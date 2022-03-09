// Copyright (C) 2015 Garth N. Wells
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
#include <xtensor/xio.hpp>
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

  std::array<std::string, 2> lagrange_identities = {"Q", "Lagrange"};
  std::array<std::string, 3> nedelec_identities
      = {"Nedelec 1st kind H(curl)", "RTCE", "NCE"};
  auto e0 = V0.element();
  std::string fam0 = e0->family();
  if (std::find(nedelec_identities.begin(), nedelec_identities.end(), fam0)
      == nedelec_identities.end())
  {
    throw std::runtime_error(
        "Output space has to be a Nedelec (first kind) function space.");
  }
  auto e1 = V1.element();
  std::string fam1 = e1->family();
  if (std::find(lagrange_identities.begin(), lagrange_identities.end(), fam1)
      == lagrange_identities.end())
  {
    throw std::runtime_error(
        "Output space has to be a Lagrange function space.");
  }

  // Tabulate Lagrange space at H(curl) interpolation points
  const xt::xtensor<double, 2> X = e0->interpolation_points();
  const int bs_1 = e1->block_size();
  assert(bs_1 == 1);
  const int ndofs_cell_1 = e1->space_dimension() / bs_1;
  const int ref_vs_1 = e1->reference_value_size();
  assert(ref_vs_1 == 1);
  const int tdim = mesh->topology().dim();
  std::array<std::size_t, 4> t_shape
      = {(std::size_t)tdim + 1, X.shape(0), (std::size_t)ndofs_cell_1,
         (std::size_t)ref_vs_1};
  xt::xtensor<double, 4> l_basis(t_shape);
  e1->tabulate(l_basis, X, 1);
  auto l_dphi
      = xt::view(l_basis, xt::xrange(1, tdim + 1), xt::all(), xt::all(), 0);
  auto dphi = xt::reshape_view(l_dphi, {tdim * t_shape[1], t_shape[2]});

  // Compute local element interpolation matrix
  const int bs_0 = e0->block_size();
  assert(bs_0 == 1);
  const int ndofs_cell_0 = e0->space_dimension() / bs_0;
  xt::xtensor<double, 2> Ae = xt::zeros<double>({ndofs_cell_0, ndofs_cell_1});
  const xt::xtensor<double, 2> Pi = e0->interpolation_operator();
  math::dot(Pi, dphi, Ae);

  // Get inverse transform
  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_inverse_dof_transform
      = e0->get_dof_transformation_function<T>(true, true, false);
  mesh->topology_mutable().create_entity_permutations();
  xtl::span<const std::uint32_t> cell_info
      = xtl::span(mesh->topology().get_cell_permutation_info());
  auto dofmap0 = V0.dofmap();
  auto dofmap1 = V1.dofmap();
  // Create element kernel
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
