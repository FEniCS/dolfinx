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
#include <xtl/xspan.hpp>

namespace dolfinx::fem
{

/// This function class computes the sparsity pattern for discrete gradient
/// operators (matrices) that map derivatives of finite element functions into
/// other finite element spaces.
///
/// Build the sparsity for the discrete gradient operator A that takes a
/// \f$w \in H^p\f$ (p-th order nodal Lagrange) to \f$v \in H(curl)\f$
/// (p-th order Nedelec first kind), i.e. v = Aw. V0 is the H(curl) space,
/// and V1 is the Lagrange space.
///
/// @param[in] V0 A p-th order Nedelec (first kind) space
/// @param[in] V1 A p-th order Lagrange space
/// @return The sparsity pattern
la::SparsityPattern
create_sparsity_discrete_gradient(const fem::FunctionSpace& V0,
                                  const fem::FunctionSpace& V1);

/// @todo Improve documentation
/// This function class computes discrete gradient operators (matrices)
/// that map derivatives of finite element functions into other finite
/// element spaces. An example of where discrete gradient operators are
/// required is the creation of algebraic multigrid solvers for H(curl)
/// and H(div) problems.
///
/// @warning This function is highly experimental and likely to change
/// or be replaced or be removed
///
/// Build the discrete gradient operator A that takes a
/// \f$w \in H^1\f$ (P1, nodal Lagrange) to \f$v \in H(curl)\f$
/// (lowest order Nedelec), i.e. v = Aw. V0 is the H(curl) space,
/// and V1 is the P1 Lagrange space.
///
/// @param[in] mat_set A function (or lambda capture) to set values in a matrix
/// @param[in] V0 A H(curl) space
/// @param[in] V1 A P1 Lagrange space
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

  // Check that first input space is Nedelec (first kind) or equivalent space on
  // quad/hex
  std::array<std::string, 3> nedelec_identities
      = {"Nedelec 1st kind H(curl)", "RTCE", "NCE"};
  auto e0 = V0.element();
  if (std::string fam0 = e0->family();
      std::find(nedelec_identities.begin(), nedelec_identities.end(), fam0)
      == nedelec_identities.end())
  {
    throw std::runtime_error(
        "Output space has to be a Nedelec (first kind) function space.");
  }

  // Check that second input space is a Lagrange space
  std::array<std::string, 2> lagrange_identities = {"Q", "Lagrange"};
  auto e1 = V1.element();
  if (std::string fam1 = e1->family();
      std::find(lagrange_identities.begin(), lagrange_identities.end(), fam1)
      == lagrange_identities.end())
  {
    throw std::runtime_error(
        "Input space has to be a Lagrange function space.");
  }

  // Get H(curl) interpolation points
  const xt::xtensor<double, 2> X = e0->interpolation_points();

  // Tabulate first order derivatives of Lagrange space at H(curl) interpolation
  // points
  const auto bs_1 = (std::size_t)e1->block_size();
  assert(bs_1 == 1);
  const auto ndofs_cell_1 = (std::size_t)e1->space_dimension() / bs_1;
  const auto ref_vs_1 = (std::size_t)e1->reference_value_size();
  assert(ref_vs_1 == 1);
  const auto tdim = (std::size_t)mesh->topology().dim();
  std::array<std::size_t, 4> t_shape
      = {tdim + 1, X.shape(0), ndofs_cell_1, ref_vs_1};
  xt::xtensor<double, 4> lagrange_reference_basis(t_shape);
  e1->tabulate(lagrange_reference_basis, X, 1);

  // Reshape lagrange basis derivatives as a matrix of shape (tdim * num_points,
  // num_dofs_per_cell)
  auto lagrange_reference_basis_derivatives
      = xt::view(lagrange_reference_basis, xt::xrange(1, (int)tdim + 1),
                 xt::all(), xt::all(), 0);
  auto dphi = xt::reshape_view(lagrange_reference_basis_derivatives,
                               {tdim * t_shape[1], t_shape[2]});

  // Get the eleemnt interpolation matrix
  const auto bs_0 = (std::size_t)e0->block_size();
  assert(bs_0 == 1);
  const auto ndofs_cell_0 = (std::size_t)e0->space_dimension() / bs_0;
  xt::xtensor<T, 2> Ae = xt::zeros<double>({ndofs_cell_0, ndofs_cell_1});
  const xt::xtensor<double, 2> Pi = e0->interpolation_operator();
  math::dot(Pi, dphi, Ae);

  // Get inverse DOF transform function
  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_inverse_dof_transform
      = e0->get_dof_transformation_function<T>(true, true, false);
  // Generate cell permutations
  mesh->topology_mutable().create_entity_permutations();
  xtl::span<const std::uint32_t> cell_info
      = xtl::span(mesh->topology().get_cell_permutation_info());

  // Create element kernel
  auto dofmap0 = V0.dofmap();
  auto dofmap1 = V1.dofmap();
  auto kernel = [&dofmap0, &dofmap1, &apply_inverse_dof_transform, &cell_info,
                 ndofs_cell_1](auto mat_set, auto Ae, const auto cell)
  {
    const xtl::span<T> _Ae(Ae);
    apply_inverse_dof_transform(_Ae, cell_info, cell, ndofs_cell_1);
    auto rows = dofmap0->cell_dofs(cell);
    auto cols = dofmap1->cell_dofs(cell);
    mat_set(rows, cols, _Ae);
  };

  // Insert local interpolation matrix for each cell
  auto cell_map = mesh->topology().index_map(tdim);
  assert(cell_map);
  std::size_t num_cells = cell_map->size_local();
  for (std::size_t i = 0; i < num_cells; i++)
    kernel(mat_set, Ae, i);
}
} // namespace dolfinx::fem
