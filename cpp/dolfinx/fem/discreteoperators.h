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
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::fem
{

/// @brief Assemble a discrete gradient operator.
///
/// The discrete gradient operator \f$A\f$ interpolates the gradient of
/// a Lagrange finite element function in \f$V_0 \subset H^1\f$ into a
/// Nédélec (first kind) space \f$V_1 \subset H({\rm curl})\f$, i.e.
/// \f$\nabla V_0 \rightarrow V_1\f$. If \f$u_0\f$ is the
/// degree-of-freedom vector associated with \f$V_0\f$, the hen
/// \f$u_1=Au_0\f$ where \f$u_1\f$ is the degrees-of-freedom vector for
/// interpolating function in the \f$H({\rm curl})\f$ space. An example
/// of where discrete gradient operators are used is the creation of
/// algebraic multigrid solvers for \f$H({\rm curl})\f$  and
/// \f$H({\rm div})\f$ problems.
///
/// @note The sparsity pattern for a discrete operator can be
/// initialised using sparsitybuild::cells.
///
/// @warning This function relies on the user supplying appropriate
/// input and output spaces. See parameter descriptions.
///
/// @param[in] V0 A Lagrange space to interpolate the gradient from
/// @param[in] V1 A Nédélec (first kind) space to interpolate into
/// @param[in] mat_set A functor that sets values in a matrix
template <typename T, typename U>
void assemble_discrete_gradient(const fem::FunctionSpace& V0,
                                const fem::FunctionSpace& V1, U&& mat_set)
{
  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V1.mesh();
  assert(mesh);

  // Check spaces
  std::shared_ptr<const FiniteElement> e0 = V0.element();
  assert(e0);
  if (e0->map_type() != basix::maps::type::identity)
    throw std::runtime_error("Wrong finite element space for V0.");
  if (e0->block_size() != 1)
    throw std::runtime_error("Block size is greather than 1 for V0.");
  if (e0->reference_value_size() != 1)
    throw std::runtime_error("Wrong value size for V0.");

  std::shared_ptr<const FiniteElement> e1 = V1.element();
  assert(e1);
  if (e1->map_type() != basix::maps::type::covariantPiola)
    throw std::runtime_error("Wrong finite element space for V1.");
  if (e1->block_size() != 1)
    throw std::runtime_error("Block size is greather than 1 for V1.");

  // Get V0 (H(curl)) space interpolation points
  const xt::xtensor<double, 2> X = e1->interpolation_points();

  // Tabulate first order derivatives of Lagrange space at H(curl)
  // interpolation points
  const int ndofs0 = e0->space_dimension();
  const int tdim = mesh->topology().dim();
  xt::xtensor<double, 4> phi0
      = xt::empty<double>({tdim + 1, int(X.shape(0)), ndofs0, 1});
  e0->tabulate(phi0, X, 1);

  // Reshape lagrange basis derivatives as a matrix of shape (tdim *
  // num_points, num_dofs_per_cell)
  auto dphi0 = xt::view(phi0, xt::xrange(std::size_t(1), phi0.shape(0)),
                        xt::all(), xt::all(), 0);
  auto dphi_reshaped
      = xt::reshape_view(dphi0, {tdim * phi0.shape(1), phi0.shape(2)});

  // Get inverse DOF transform function
  auto apply_inverse_dof_transform
      = e1->get_dof_transformation_function<T>(true, true, false);

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
  std::vector<T> A(e1->space_dimension() * ndofs0);
  {
    auto _A = xt::adapt(A, std::vector<int>{e1->space_dimension(), ndofs0});
    const xt::xtensor<double, 2> Pi = e1->interpolation_operator();
    math::dot(Pi, dphi_reshaped, _A);
  }

  // Insert local interpolation matrix for each cell
  auto cell_map = mesh->topology().index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  std::vector<T> Ae(A.size());
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    std::copy(A.cbegin(), A.cend(), Ae.begin());
    apply_inverse_dof_transform(Ae, cell_info, c, ndofs0);
    mat_set(dofmap1->cell_dofs(c), dofmap0->cell_dofs(c), Ae);
  }
}
} // namespace dolfinx::fem
