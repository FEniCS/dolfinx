// Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <span>
#include <vector>

namespace dolfinx::fem
{

/// @brief Assemble a discrete gradient operator.
///
/// The discrete gradient operator \f$A\f$ interpolates the gradient of
/// a Lagrange finite element function in \f$V_0 \subset H^1\f$ into a
/// Nédélec (first kind) space \f$V_1 \subset H({\rm curl})\f$, i.e.
/// \f$\nabla V_0 \rightarrow V_1\f$. If \f$u_0\f$ is the
/// degree-of-freedom vector associated with \f$V_0\f$, then
/// \f$u_1=Au_0\f$ where \f$u_1\f$ is the degrees-of-freedom vector for
/// interpolating function in the \f$H({\rm curl})\f$ space. An example
/// of where discrete gradient operators are used is the creation of
/// algebraic multigrid solvers for \f$H({\rm curl})\f$ and \f$H({\rm
/// div})\f$ problems.
///
/// @note The sparsity pattern for a discrete operator can be
/// initialised using sparsitybuild::cells. The space `V1` should be
/// used for the rows of the sparsity pattern, `V0` for the columns.
///
/// @warning This function relies on the user supplying appropriate
/// input and output spaces. See parameter descriptions.
///
/// @param[in] topology Mesh topology
/// @param[in] V0 Lagrange element and dofmap for corresponding space to
/// interpolate the gradient from.
/// @param[in] V1 Nédélec (first kind) element and dofmap for
/// corresponding space to interpolate into.
/// @param[in] mat_set A functor that sets values in a matrix
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void discrete_gradient(mesh::Topology& topology,
                       std::pair<std::reference_wrapper<const FiniteElement<U>>,
                                 std::reference_wrapper<const DofMap>>
                           V0,
                       std::pair<std::reference_wrapper<const FiniteElement<U>>,
                                 std::reference_wrapper<const DofMap>>
                           V1,
                       auto&& mat_set)
{
  auto& e0 = V0.first.get();
  const DofMap& dofmap0 = V0.second.get();
  auto& e1 = V1.first.get();
  const DofMap& dofmap1 = V1.second.get();

  using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

  // Check elements
  if (e0.map_type() != basix::maps::type::identity)
    throw std::runtime_error("Wrong finite element space for V0.");
  if (e0.block_size() != 1)
    throw std::runtime_error("Block size is greater than 1 for V0.");
  if (e0.reference_value_size() != 1)
    throw std::runtime_error("Wrong value size for V0.");

  if (e1.map_type() != basix::maps::type::covariantPiola)
    throw std::runtime_error("Wrong finite element space for V1.");
  if (e1.block_size() != 1)
    throw std::runtime_error("Block size is greater than 1 for V1.");

  // Get V0 (H(curl)) space interpolation points
  const auto [X, Xshape] = e1.interpolation_points();

  // Tabulate first order derivatives of Lagrange space at H(curl)
  // interpolation points
  const int ndofs0 = e0.space_dimension();
  const int tdim = topology.dim();
  std::vector<U> phi0_b((tdim + 1) * Xshape[0] * ndofs0 * 1);
  cmdspan4_t phi0(phi0_b.data(), tdim + 1, Xshape[0], ndofs0, 1);
  e0.tabulate(phi0_b, X, Xshape, 1);

  // Reshape lagrange basis derivatives as a matrix of shape (tdim *
  // num_points, num_dofs_per_cell)
  cmdspan2_t dphi_reshaped(
      phi0_b.data() + phi0.extent(3) * phi0.extent(2) * phi0.extent(1),
      tdim * phi0.extent(1), phi0.extent(2));

  // Get inverse DOF transform function
  auto apply_inverse_dof_transform = e1.template dof_transformation_fn<T>(
      doftransform::inverse_transpose, false);

  // Generate cell permutations
  topology.create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = topology.get_cell_permutation_info();

  // Create element kernel function

  // Build the element interpolation matrix
  std::vector<T> Ab(e1.space_dimension() * ndofs0);
  {
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        A(Ab.data(), e1.space_dimension(), ndofs0);
    const auto [Pi, shape] = e1.interpolation_operator();
    cmdspan2_t _Pi(Pi.data(), shape);
    math::dot(_Pi, dphi_reshaped, A);
  }

  // Insert local interpolation matrix for each cell
  auto cell_map = topology.index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  std::vector<T> Ae(Ab.size());
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    std::ranges::copy(Ab, Ae.begin());
    apply_inverse_dof_transform(Ae, cell_info, c, ndofs0);
    mat_set(dofmap1.cell_dofs(c), dofmap0.cell_dofs(c), Ae);
  }
}

/// @brief Assemble an interpolation operator matrix.
///
/// The interpolation operator \f$A\f$ interpolates a function in the
/// space \f$V_0\f$ into a space \f$V_1\f$. If \f$u_0\f$ is the
/// degree-of-freedom vector associated with \f$V_0\f$, then the
/// degree-of-freedom vector \f$u_1\f$ for the interpolated function in
/// \f$V_1\f$ is given by \f$u_1=Au_0\f$.
///
/// @note The sparsity pattern for a discrete operator can be
/// initialised using sparsitybuild::cells. The space `V1` should be
/// used for the rows of the sparsity pattern, `V0` for the columns.
///
/// @param[in] V0 Space to interpolate from.
/// @param[in] V1 Space to interpolate to.
/// @param[in] mat_set Functor that sets values in a matrix.
template <dolfinx::scalar T, std::floating_point U>
void interpolation_matrix(const FunctionSpace<U>& V0,
                          const FunctionSpace<U>& V1, auto&& mat_set)
{
  // Get mesh
  auto mesh = V0.mesh();
  assert(mesh);

  // Mesh dims
  const int tdim = mesh->topology()->dim();
  const int gdim = mesh->geometry().dim();

  // Get elements
  std::shared_ptr<const FiniteElement<U>> e0 = V0.element();
  assert(e0);
  std::shared_ptr<const FiniteElement<U>> e1 = V1.element();
  assert(e1);

  std::span<const std::uint32_t> cell_info;
  if (e1->needs_dof_transformations() or e0->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap0 = V0.dofmap();
  assert(dofmap0);
  auto dofmap1 = V1.dofmap();
  assert(dofmap1);

  // Get block sizes and dof transformation operators
  const int bs0 = e0->block_size();
  const int bs1 = e1->block_size();
  auto apply_dof_transformation0
      = e0->template dof_transformation_fn<U>(doftransform::standard, false);
  auto apply_inverse_dof_transform1 = e1->template dof_transformation_fn<T>(
      doftransform::inverse_transpose, false);

  // Get sizes of elements
  const std::size_t space_dim0 = e0->space_dimension();
  const std::size_t space_dim1 = e1->space_dimension();
  const std::size_t dim0 = space_dim0 / bs0;
  const std::size_t value_size_ref0 = e0->reference_value_size();
  const std::size_t value_size0 = V0.element()->reference_value_size();
  const std::size_t value_size1 = V1.element()->reference_value_size();

  // Get geometry data
  const CoordinateElement<U>& cmap = mesh->geometry().cmap();
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const U> x_g = mesh->geometry().x();

  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using cmdspan3_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>;
  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  using mdspan3_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>;

  // Evaluate coordinate map basis at reference interpolation points
  const auto [X, Xshape] = e1->interpolation_points();
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, Xshape[0]);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  cmap.tabulate(1, X, Xshape, phi_b);

  // Evaluate V0 basis functions at reference interpolation points for V1
  std::vector<U> basis_derivatives_reference0_b(Xshape[0] * dim0
                                                * value_size_ref0);
  cmdspan4_t basis_derivatives_reference0(basis_derivatives_reference0_b.data(),
                                          1, Xshape[0], dim0, value_size_ref0);
  e0->tabulate(basis_derivatives_reference0_b, X, Xshape, 0);

  // Clamp values
  std::ranges::transform(
      basis_derivatives_reference0_b, basis_derivatives_reference0_b.begin(),
      [atol = 1e-14](auto x) { return std::abs(x) < atol ? 0.0 : x; });

  // Create working arrays
  std::vector<U> basis_reference0_b(Xshape[0] * dim0 * value_size_ref0);
  mdspan3_t basis_reference0(basis_reference0_b.data(), Xshape[0], dim0,
                             value_size_ref0);
  std::vector<U> J_b(Xshape[0] * gdim * tdim);
  mdspan3_t J(J_b.data(), Xshape[0], gdim, tdim);
  std::vector<U> K_b(Xshape[0] * tdim * gdim);
  mdspan3_t K(K_b.data(), Xshape[0], tdim, gdim);
  std::vector<U> detJ(Xshape[0]);
  std::vector<U> det_scratch(2 * tdim * gdim);

  // Get the interpolation operator (matrix) `Pi` that maps a function
  // evaluated at the interpolation points to the element degrees of
  // freedom, i.e. dofs = Pi f_x
  const auto [_Pi_1, pi_shape] = e1->interpolation_operator();
  cmdspan2_t Pi_1(_Pi_1.data(), pi_shape);

  bool interpolation_ident = e1->interpolation_ident();

  using u_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using U_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using J_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using K_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  auto push_forward_fn0
      = e0->basix_element().template map_fn<u_t, U_t, J_t, K_t>();

  // Basis values of Lagrange space unrolled for block size
  // (num_quadrature_points, Lagrange dof, value_size)
  std::vector<U> basis_values_b(Xshape[0] * bs0 * dim0
                                * V1.element()->value_size());
  mdspan3_t basis_values(basis_values_b.data(), Xshape[0], bs0 * dim0,
                         V1.element()->value_size());
  std::vector<U> mapped_values_b(Xshape[0] * bs0 * dim0
                                 * V1.element()->value_size());
  mdspan3_t mapped_values(mapped_values_b.data(), Xshape[0], bs0 * dim0,
                          V1.element()->value_size());

  auto pull_back_fn1
      = e1->basix_element().template map_fn<u_t, U_t, K_t, J_t>();

  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> basis0_b(Xshape[0] * dim0 * value_size0);
  mdspan3_t basis0(basis0_b.data(), Xshape[0], dim0, value_size0);

  // Buffers
  std::vector<T> Ab(space_dim0 * space_dim1);
  std::vector<T> local1(space_dim1);

  // Iterate over mesh and interpolate on each cell
  auto cell_map = mesh->topology()->index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Get cell geometry (coordinate dofs)
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];
    }

    // Compute Jacobians and reference points for current cell
    std::ranges::fill(J_b, 0);
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          phi, std::pair(1, tdim + 1), p,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      cmap.compute_jacobian(dphi, coord_dofs, _J);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      cmap.compute_jacobian_inverse(_J, _K);
      detJ[p] = cmap.compute_jacobian_determinant(_J, det_scratch);
    }

    // Copy evaluated basis on reference, apply DOF transformations, and
    // push forward to physical element
    for (std::size_t k0 = 0; k0 < basis_reference0.extent(0); ++k0)
      for (std::size_t k1 = 0; k1 < basis_reference0.extent(1); ++k1)
        for (std::size_t k2 = 0; k2 < basis_reference0.extent(2); ++k2)
          basis_reference0(k0, k1, k2)
              = basis_derivatives_reference0(0, k0, k1, k2);
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      apply_dof_transformation0(
          std::span(basis_reference0.data_handle() + p * dim0 * value_size_ref0,
                    dim0 * value_size_ref0),
          cell_info, c, value_size_ref0);
    }

    for (std::size_t p = 0; p < basis0.extent(0); ++p)
    {
      auto _u = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis0, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _U = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis_reference0, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      push_forward_fn0(_u, _U, _J, detJ[p], _K);
    }

    // Unroll basis function for input space for block size
    for (std::size_t p = 0; p < Xshape[0]; ++p)
      for (std::size_t i = 0; i < dim0; ++i)
        for (std::size_t j = 0; j < value_size0; ++j)
          for (int k = 0; k < bs0; ++k)
            basis_values(p, i * bs0 + k, j * bs0 + k) = basis0(p, i, j);

    // Pull back the physical values to the reference of output space
    for (std::size_t p = 0; p < basis_values.extent(0); ++p)
    {
      auto _u = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis_values, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _U = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          mapped_values, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      pull_back_fn1(_U, _u, _K, 1.0 / detJ[p], _J);
    }

    // Apply interpolation matrix to basis values of V0 at the
    // interpolation points of V1
    if (interpolation_ident)
    {
      MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
          T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
          A(Ab.data(), Xshape[0], V1.element()->value_size(), space_dim0);
      for (std::size_t i = 0; i < mapped_values.extent(0); ++i)
        for (std::size_t j = 0; j < mapped_values.extent(1); ++j)
          for (std::size_t k = 0; k < mapped_values.extent(2); ++k)
            A(i, k, j) = mapped_values(i, j, k);
    }
    else
    {
      for (std::size_t i = 0; i < mapped_values.extent(1); ++i)
      {
        auto values = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            mapped_values, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, i,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        impl::interpolation_apply(Pi_1, values, std::span(local1), bs1);
        for (std::size_t j = 0; j < local1.size(); j++)
          Ab[space_dim0 * j + i] = local1[j];
      }
    }

    apply_inverse_dof_transform1(Ab, cell_info, c, space_dim0);
    mat_set(dofmap1->cell_dofs(c), dofmap0->cell_dofs(c), Ab);
  }
}

} // namespace dolfinx::fem
