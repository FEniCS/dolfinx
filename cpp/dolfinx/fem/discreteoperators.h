// Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/math.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::fem
{

/// @brief Create a sparsity pattern for a discrete gradient
/// operator.
///
/// @warning This function relies on the user supplying appropriate
/// input and output spaces. See parameter descriptions.
///
/// @param[in] V0 A q-th order Nedelec (first kind) space
/// @param[in] V1 A p-th order Lagrange space
/// @param[in] V0 A degree q Nedelec (first kind) space
/// @param[in] V1 A degere p Lagrange space
la::SparsityPattern
create_sparsity_discrete_gradient(const fem::FunctionSpace& V0,
                                  const fem::FunctionSpace& V1);

/// @brief Assemble a discrete gradient operator.
///
/// Build the discrete gradient operator \f$A\f$ that takes a
/// \f$w \in H^p\f$ (p-th order nodal Lagrange) to \f$v \in H(curl)\f$
/// (q-th order Nedelec first kind), i.e. v = Aw. V0 is the H(curl) space,
/// and V1 is the Lagrange space.
///
/// An example of where discrete gradient operators are required is the
/// creation of algebraic multigrid solvers for H(curl) and H(div)
/// problems.
///
/// @warning This function relies on the user supplying appropriate
/// input and output spaces. See parameter descriptions.
///
/// @param[in] mat_set A functor that sets values in a matrix
/// @param[in] V0 A degree q Nedelec (first kind) space
/// @param[in] V1 A degere p Lagrange space
template <typename T>
void assemble_discrete_gradient(
    const std::function<int(const xtl::span<const std::int32_t>&,
                            const xtl::span<const std::int32_t>&,
                            const xtl::span<const T>&)>& mat_set,
    const fem::FunctionSpace& V0, const fem::FunctionSpace& V1)
{
  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);

  // Check spaces
  std::shared_ptr<const FiniteElement> e0 = V0.element();
  assert(e0);
  if (e0->map_type() != basix::maps::type::covariantPiola)
    throw std::runtime_error("Wrong finite element space for V0.");
  if (e0->block_size() != 1)
    throw std::runtime_error("Block size is greather than 1 for V0.");

  std::shared_ptr<const FiniteElement> e1 = V1.element();
  assert(e1);
  if (e1->map_type() != basix::maps::type::identity)
    throw std::runtime_error("Wrong finite element space for V1.");
  if (e1->block_size() != 1)
    throw std::runtime_error("Block size is greather than 1 for V1.");
  if (e1->reference_value_size() != 1)
    throw std::runtime_error("Wrong value size for V1.");

  // Get V1 (H(curl)) space interpolation points
  const xt::xtensor<double, 2> X = e0->interpolation_points();

  // Tabulate first order derivatives of Lagrange space at H(curl) interpolation
  // points
  const int ndofs1 = e1->space_dimension();
  const int tdim = mesh->topology().dim();
  xt::xtensor<double, 4> phi1
      = xt::empty<double>({tdim + 1, int(X.shape(0)), ndofs1, 1});
  e1->tabulate(phi1, X, 1);

  // Reshape lagrange basis derivatives as a matrix of shape (tdim * num_points,
  // num_dofs_per_cell)
  auto dphi1 = xt::view(phi1, xt::xrange(std::size_t(1), phi1.shape(0)),
                        xt::all(), xt::all(), 0);
  auto dphi_reshaped
      = xt::reshape_view(dphi1, {tdim * phi1.shape(1), phi1.shape(2)});

  // Get inverse DOF transform function
  auto apply_inverse_dof_transform
      = e0->get_dof_transformation_function<T>(true, true, false);

  // Generate cell permutations
  mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Create element kernel function
  std::shared_ptr<const DofMap> dofmap0 = V0.dofmap();
  assert(dofmap0);
  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
  assert(dofmap1);

  // Build the element interpolation matrix
  std::vector<T> A0(e0->space_dimension() * ndofs1);
  {
    auto _A0 = xt::adapt(A0, std::vector<int>{e0->space_dimension(), ndofs1});
    const xt::xtensor<double, 2> Pi = e0->interpolation_operator();
    math::dot(Pi, dphi_reshaped, _A0);
  }

  // Insert local interpolation matrix for each cell
  auto cell_map = mesh->topology().index_map(tdim);
  assert(cell_map);
  std::size_t num_cells = cell_map->size_local();
  std::vector<T> Ae(A0.size());
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    std::copy(A0.cbegin(), A0.cend(), Ae.begin());
    apply_inverse_dof_transform(Ae, cell_info, c, ndofs1);
    mat_set(dofmap0->cell_dofs(c), dofmap1->cell_dofs(c), Ae);
  }
}
} // namespace dolfinx::fem
