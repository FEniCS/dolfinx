// Copyright (C) 2020-2021 Garth N. Wells, Igor A. Baratta
// and Jørgen S.Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

namespace dolfinx::fem
{
template <typename T>
class Function;

namespace impl
{
/// @brief Apply interpolation operator Pi to data to evaluate the dof
/// coefficients
/// @param[in] Pi The interpolation matrix (shape = (num dofs,
/// num_points * value_size))
/// @param[in] data Function evaluations, by point, e.g. (f0(x0),
/// f1(x0), f0(x1), f1(x1), ...)
/// @param[out] coeffs The degrees of freedom to compute
/// @param[in] bs The block size
template <typename U, typename V, typename T>
void interpolation_apply(const U& Pi, const V& data, std::span<T> coeffs,
                         int bs)
{
  static_assert(U::rank() == 2, "Must be rank 2");
  static_assert(V::rank() == 2, "Must be rank 2");

  // Compute coefficients = Pi * x (matrix-vector multiply)
  if (bs == 1)
  {
    assert(data.extent(0) * data.extent(1) == Pi.extent(1));
    for (std::size_t i = 0; i < Pi.extent(0); ++i)
    {
      coeffs[i] = 0.0;
      for (std::size_t k = 0; k < data.extent(1); ++k)
        for (std::size_t j = 0; j < data.extent(0); ++j)
          coeffs[i] += Pi(i, k * data.extent(0) + j) * data(j, k);
    }
  }
  else
  {
    const std::size_t cols = Pi.extent(1);
    assert(data.extent(0) == Pi.extent(1));
    assert(data.extent(1) == bs);
    for (int k = 0; k < bs; ++k)
    {
      for (std::size_t i = 0; i < Pi.extent(0); ++i)
      {
        T acc = 0;
        for (std::size_t j = 0; j < cols; ++j)
          acc += Pi(i, j) * data(j, k);
        coeffs[bs * i + k] = acc;
      }
    }
  }
}

/// Interpolate from one finite element Function to another on the same
/// mesh. The function is for cases where the finite element basis
/// functions are mapped in the same way, e.g. both use the same Piola
/// map.
/// @param[out] u1 The function to interpolate to
/// @param[in] u0 The function to interpolate from
/// @param[in] cells The cells to interpolate on
/// @pre The functions `u1` and `u0` must share the same mesh and the
/// elements must share the same basis function map. Neither is checked
/// by the function.
template <typename T>
void interpolate_same_map(Function<T>& u1, const Function<T>& u0,
                          std::span<const std::int32_t> cells)
{
  auto V0 = u0.function_space();
  assert(V0);
  auto V1 = u1.function_space();
  assert(V1);
  auto mesh = V0->mesh();
  assert(mesh);

  std::shared_ptr<const FiniteElement> element0 = V0->element();
  assert(element0);
  std::shared_ptr<const FiniteElement> element1 = V1->element();
  assert(element1);

  const int tdim = mesh->topology().dim();
  auto map = mesh->topology().index_map(tdim);
  assert(map);
  std::span<T> u1_array = u1.x()->mutable_array();
  std::span<const T> u0_array = u0.x()->array();

  std::span<const std::uint32_t> cell_info;
  if (element1->needs_dof_transformations()
      or element0->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap1 = V1->dofmap();
  auto dofmap0 = V0->dofmap();

  // Get block sizes and dof transformation operators
  const int bs1 = dofmap1->bs();
  const int bs0 = dofmap0->bs();
  auto apply_dof_transformation
      = element0->get_dof_transformation_function<T>(false, true, false);
  auto apply_inverse_dof_transform
      = element1->get_dof_transformation_function<T>(true, true, false);

  // Create working array
  std::vector<T> local0(element0->space_dimension());
  std::vector<T> local1(element1->space_dimension());

  // Create interpolation operator
  const auto [i_m, im_shape]
      = element1->create_interpolation_operator(*element0);

  // Iterate over mesh and interpolate on each cell
  for (auto c : cells)
  {
    std::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(c);
    for (std::size_t i = 0; i < dofs0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        local0[bs0 * i + k] = u0_array[bs0 * dofs0[i] + k];

    apply_dof_transformation(local0, cell_info, c, 1);

    // FIXME: Get compile-time ranges from Basix
    // Apply interpolation operator
    std::fill(local1.begin(), local1.end(), 0);
    for (std::size_t i = 0; i < im_shape[0]; ++i)
      for (std::size_t j = 0; j < im_shape[1]; ++j)
        local1[i] += i_m[im_shape[1] * i + j] * local0[j];

    apply_inverse_dof_transform(local1, cell_info, c, 1);

    std::span<const std::int32_t> dofs1 = dofmap1->cell_dofs(c);
    for (std::size_t i = 0; i < dofs1.size(); ++i)
      for (int k = 0; k < bs1; ++k)
        u1_array[bs1 * dofs1[i] + k] = local1[bs1 * i + k];
  }
}

/// Interpolate from one finite element Function to another on the same
/// mesh. The function is for cases where the finite element basis
/// functions for the two elements are mapped differently, e.g. one may
/// be Piola mapped and the other with a standard isoparametric map.
/// @param[out] u1 The function to interpolate to
/// @param[in] u0 The function to interpolate from
/// @param[in] cells The cells to interpolate on
/// @pre The functions `u1` and `u0` must share the same mesh. This is
/// not checked by the function.
template <typename T>
void interpolate_nonmatching_maps(Function<T>& u1, const Function<T>& u0,
                                  std::span<const std::int32_t> cells)
{
  // Get mesh
  auto V0 = u0.function_space();
  assert(V0);
  auto mesh = V0->mesh();
  assert(mesh);

  // Mesh dims
  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();

  // Get elements
  auto V1 = u1.function_space();
  assert(V1);
  std::shared_ptr<const FiniteElement> element0 = V0->element();
  assert(element0);
  std::shared_ptr<const FiniteElement> element1 = V1->element();
  assert(element1);

  std::span<const std::uint32_t> cell_info;
  if (element1->needs_dof_transformations()
      or element0->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap0 = V0->dofmap();
  auto dofmap1 = V1->dofmap();

  const auto [X, Xshape] = element1->interpolation_points();

  // Get block sizes and dof transformation operators
  const int bs0 = element0->block_size();
  const int bs1 = element1->block_size();
  const auto apply_dof_transformation0
      = element0->get_dof_transformation_function<double>(false, false, false);
  const auto apply_inverse_dof_transform1
      = element1->get_dof_transformation_function<T>(true, true, false);

  // Get sizes of elements
  const std::size_t dim0 = element0->space_dimension() / bs0;
  const std::size_t value_size_ref0 = element0->reference_value_size() / bs0;
  const std::size_t value_size0 = element0->value_size() / bs0;

  // Get geometry data
  const CoordinateElement& cmap = mesh->geometry().cmap();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const double> x_g = mesh->geometry().x();

  namespace stdex = std::experimental;
  using cmdspan2_t
      = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using cmdspan4_t
      = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
  using mdspan3T_t = stdex::mdspan<T, stdex::dextents<std::size_t, 3>>;

  // Evaluate coordinate map basis at reference interpolation points
  const std::array<std::size_t, 4> phi_shape
      = cmap.tabulate_shape(1, Xshape[0]);
  std::vector<double> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  cmap.tabulate(1, X, Xshape, phi_b);

  // Evaluate v basis functions at reference interpolation points
  const auto [_basis_derivatives_reference0, b0shape]
      = element0->tabulate(X, Xshape, 0);
  cmdspan4_t basis_derivatives_reference0(_basis_derivatives_reference0.data(),
                                          b0shape);

  // Create working arrays
  std::vector<T> local1(element1->space_dimension());
  std::vector<T> coeffs0(element0->space_dimension());

  std::vector<double> basis0_b(Xshape[0] * dim0 * value_size0);
  mdspan3_t basis0(basis0_b.data(), Xshape[0], dim0, value_size0);

  std::vector<double> basis_reference0_b(Xshape[0] * dim0 * value_size_ref0);
  mdspan3_t basis_reference0(basis_reference0_b.data(), Xshape[0], dim0,
                             value_size_ref0);

  std::vector<T> values0_b(Xshape[0] * 1 * element1->value_size());
  mdspan3T_t values0(values0_b.data(), Xshape[0], 1, element1->value_size());

  std::vector<T> mapped_values_b(Xshape[0] * 1 * element1->value_size());
  mdspan3T_t mapped_values0(mapped_values_b.data(), Xshape[0], 1,
                            element1->value_size());

  std::vector<double> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

  std::vector<double> J_b(Xshape[0] * gdim * tdim);
  mdspan3_t J(J_b.data(), Xshape[0], gdim, tdim);
  std::vector<double> K_b(Xshape[0] * tdim * gdim);
  mdspan3_t K(K_b.data(), Xshape[0], tdim, gdim);
  std::vector<double> detJ(Xshape[0]);
  std::vector<double> det_scratch(2 * gdim * tdim);

  // Get interpolation operator
  const auto [_Pi_1, pi_shape] = element1->interpolation_operator();
  cmdspan2_t Pi_1(_Pi_1.data(), pi_shape);

  using u_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using U_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using J_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using K_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  auto push_forward_fn0
      = element0->basix_element().map_fn<u_t, U_t, J_t, K_t>();

  using v_t = stdex::mdspan<const T, stdex::dextents<std::size_t, 2>>;
  using V_t = stdex::mdspan<T, stdex::dextents<std::size_t, 2>>;
  auto pull_back_fn1 = element1->basix_element().map_fn<V_t, v_t, K_t, J_t>();

  // Iterate over mesh and interpolate on each cell
  std::span<const T> array0 = u0.x()->array();
  std::span<T> array1 = u1.x()->mutable_array();
  for (auto c : cells)
  {
    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }

    // Compute Jacobians and reference points for current cell
    std::fill(J_b.begin(), J_b.end(), 0);
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      auto dphi = stdex::submdspan(phi, std::pair(1, tdim + 1), p,
                                   stdex::full_extent, 0);

      auto _J = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
      cmap.compute_jacobian(dphi, coord_dofs, _J);
      auto _K = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);
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
          std::span(basis_reference0_b.data() + p * dim0 * value_size_ref0,
                    dim0 * value_size_ref0),
          cell_info, c, value_size_ref0);
    }

    for (std::size_t i = 0; i < basis0.extent(0); ++i)
    {
      auto _u
          = stdex::submdspan(basis0, i, stdex::full_extent, stdex::full_extent);
      auto _U = stdex::submdspan(basis_reference0, i, stdex::full_extent,
                                 stdex::full_extent);
      auto _K = stdex::submdspan(K, i, stdex::full_extent, stdex::full_extent);
      auto _J = stdex::submdspan(J, i, stdex::full_extent, stdex::full_extent);
      push_forward_fn0(_u, _U, _J, detJ[i], _K);
    }

    // Copy expansion coefficients for v into local array
    const int dof_bs0 = dofmap0->bs();
    std::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(c);
    for (std::size_t i = 0; i < dofs0.size(); ++i)
      for (int k = 0; k < dof_bs0; ++k)
        coeffs0[dof_bs0 * i + k] = array0[dof_bs0 * dofs0[i] + k];

    // Evaluate v at the interpolation points (physical space values)
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      for (int k = 0; k < bs0; ++k)
      {
        for (std::size_t j = 0; j < value_size0; ++j)
        {
          T acc = 0;
          for (std::size_t i = 0; i < dim0; ++i)
            acc += coeffs0[bs0 * i + k] * basis0(p, i, j);
          values0(p, 0, j * bs0 + k) = acc;
        }
      }
    }

    // Pull back the physical values to the u reference
    for (std::size_t i = 0; i < values0.extent(0); ++i)
    {
      auto _u = stdex::submdspan(values0, i, stdex::full_extent,
                                 stdex::full_extent);
      auto _U = stdex::submdspan(mapped_values0, i, stdex::full_extent,
                                 stdex::full_extent);
      auto _K = stdex::submdspan(K, i, stdex::full_extent, stdex::full_extent);
      auto _J = stdex::submdspan(J, i, stdex::full_extent, stdex::full_extent);
      pull_back_fn1(_U, _u, _K, 1.0 / detJ[i], _J);
    }

    auto values = stdex::submdspan(mapped_values0, stdex::full_extent, 0,
                                   stdex::full_extent);
    interpolation_apply(Pi_1, values, std::span(local1), bs1);
    apply_inverse_dof_transform1(local1, cell_info, c, 1);

    // Copy local coefficients to the correct position in u dof array
    const int dof_bs1 = dofmap1->bs();
    std::span<const std::int32_t> dofs1 = dofmap1->cell_dofs(c);
    for (std::size_t i = 0; i < dofs1.size(); ++i)
      for (int k = 0; k < dof_bs1; ++k)
        array1[dof_bs1 * dofs1[i] + k] = local1[dof_bs1 * i + k];
  }
}
} // namespace impl

/// Compute the evaluation points in the physical space at which an
/// expression should be computed to interpolate it in a finite element
/// space.
///
/// @param[in] element The element to be interpolated into
/// @param[in] mesh The domain
/// @param[in] cells Indices of the cells in the mesh to compute
/// interpolation coordinates for
/// @return The coordinates in the physical space at which to evaluate
/// an expression. The shape is (3, num_points) and storage is row-major.
std::vector<double> interpolation_coords(const FiniteElement& element,
                                         const mesh::Mesh& mesh,
                                         std::span<const std::int32_t> cells);

/// Interpolate an expression f(x) in a finite element space
///
/// @param[out] u The function to interpolate into
/// @param[in] f Evaluation of the function `f(x)` at the physical
/// points `x` given by fem::interpolation_coords. The element used in
/// fem::interpolation_coords should be the same element as associated
/// with `u`. The shape of `f` should be (value_size, num_points), with
/// row-major storage.
/// @param[in] fshape The shape of `f`.
/// @param[in] cells Indices of the cells in the mesh on which to
/// interpolate. Should be the same as the list used when calling
/// fem::interpolation_coords.
template <typename T>
void interpolate(Function<T>& u, std::span<const T> f,
                 std::array<std::size_t, 2> fshape,
                 std::span<const std::int32_t> cells)
{
  namespace stdex = std::experimental;
  using cmdspan2_t
      = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using cmdspan4_t
      = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;

  std::shared_ptr<const FiniteElement> element = u.function_space()->element();
  assert(element);
  const int element_bs = element->block_size();
  if (int num_sub = element->num_sub_elements();
      num_sub > 0 and num_sub != element_bs)
  {
    throw std::runtime_error("Cannot directly interpolate a mixed space. "
                             "Interpolate into subspaces.");
  }

  if (fshape[0] != (std::size_t)element->value_size())
    throw std::runtime_error("Interpolation data has the wrong shape/size.");

  // Get mesh
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);

  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  const std::size_t f_shape1 = f.size() / element->value_size();
  stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> _f(f.data(), fshape);

  // Get dofmap
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int dofmap_bs = dofmap->bs();

  // Loop over cells and compute interpolation dofs
  const int num_scalar_dofs = element->space_dimension() / element_bs;
  const int value_size = element->value_size() / element_bs;

  std::span<T> coeffs = u.x()->mutable_array();
  std::vector<T> _coeffs(num_scalar_dofs);

  // This assumes that any element with an identity interpolation matrix
  // is a point evaluation
  if (element->map_ident() && element->interpolation_ident())
  {
    // Point evaluation element *and* the geometric map is the identity,
    // e.g. not Piola mapped

    auto apply_inv_transpose_dof_transformation
        = element->get_dof_transformation_function<T>(true, true, true);

    // Loop over cells
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
      const std::int32_t cell = cells[c];
      std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
      for (int k = 0; k < element_bs; ++k)
      {
        // num_scalar_dofs is the number of interpolation points per
        // cell in this case (interpolation matrix is identity)
        std::copy_n(std::next(f.begin(), k * f_shape1 + c * num_scalar_dofs),
                    num_scalar_dofs, _coeffs.begin());
        apply_inv_transpose_dof_transformation(_coeffs, cell_info, cell, 1);
        for (int i = 0; i < num_scalar_dofs; ++i)
        {
          const int dof = i * element_bs + k;
          std::div_t pos = std::div(dof, dofmap_bs);
          coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
        }
      }
    }
  }
  else if (element->map_ident())
  {
    // Not a point evaluation, but the geometric map is the identity,
    // e.g. not Piola mapped

    const int element_vs = element->value_size() / element_bs;

    if (element_vs > 1 && element_bs > 1)
    {
      throw std::runtime_error(
          "Interpolation into this element not supported.");
    }

    // Get interpolation operator
    const auto [_Pi, pi_shape] = element->interpolation_operator();
    cmdspan2_t Pi(_Pi.data(), pi_shape);
    const std::size_t num_interp_points = Pi.extent(1);
    assert(Pi.extent(0) == num_scalar_dofs);

    auto apply_inv_transpose_dof_transformation
        = element->get_dof_transformation_function<T>(true, true, true);

    // Loop over cells
    std::vector<T> ref_data_b(num_interp_points);
    stdex::mdspan<T, stdex::extents<std::size_t, stdex::dynamic_extent, 1>>
        ref_data(ref_data_b.data(), num_interp_points, 1);
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
      const std::int32_t cell = cells[c];
      std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
      for (int k = 0; k < element_bs; ++k)
      {
        for (int i = 0; i < element_vs; ++i)
        {
          std::copy_n(
              std::next(f.begin(), (i + k) * f_shape1
                                       + c * num_interp_points / element_vs),
              num_interp_points / element_vs,
              std::next(ref_data_b.begin(),
                        i * num_interp_points / element_vs));
        }
        impl::interpolation_apply(Pi, ref_data, std::span(_coeffs), 1);
        apply_inv_transpose_dof_transformation(_coeffs, cell_info, cell, 1);
        for (int i = 0; i < num_scalar_dofs; ++i)
        {
          const int dof = i * element_bs + k;
          std::div_t pos = std::div(dof, dofmap_bs);
          coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
        }
      }
    }
  }
  else
  {
    // Get the interpolation points on the reference cells
    const auto [X, Xshape] = element->interpolation_points();
    if (X.empty())
    {
      throw std::runtime_error(
          "Interpolation into this space is not yet supported.");
    }

    if (_f.extent(1) != cells.size() * Xshape[0])
      throw std::runtime_error("Interpolation data has the wrong shape.");

    // Get coordinate map
    const CoordinateElement& cmap = mesh->geometry().cmap();

    // Get geometry data
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();
    const int num_dofs_g = cmap.dim();
    std::span<const double> x_g = mesh->geometry().x();

    // Create data structures for Jacobian info
    std::vector<double> J_b(Xshape[0] * gdim * tdim);
    mdspan3_t J(J_b.data(), Xshape[0], gdim, tdim);
    std::vector<double> K_b(Xshape[0] * tdim * gdim);
    mdspan3_t K(K_b.data(), Xshape[0], tdim, gdim);
    std::vector<double> detJ(Xshape[0]);
    std::vector<double> det_scratch(2 * gdim * tdim);

    std::vector<double> coord_dofs_b(num_dofs_g * gdim);
    mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

    std::vector<T> ref_data_b(Xshape[0] * 1 * value_size);
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> ref_data(
        ref_data_b.data(), Xshape[0], 1, value_size);

    std::vector<T> _vals_b(Xshape[0] * 1 * value_size);
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> _vals(
        _vals_b.data(), Xshape[0], 1, value_size);

    // Tabulate 1st derivative of shape functions at interpolation
    // coords
    std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, Xshape[0]);
    std::vector<double> phi_b(
        std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    cmdspan4_t phi(phi_b.data(), phi_shape);
    cmap.tabulate(1, X, Xshape, phi_b);
    auto dphi = stdex::submdspan(phi, std::pair(1, tdim + 1),
                                 stdex::full_extent, stdex::full_extent, 0);

    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>
        apply_inverse_transpose_dof_transformation
        = element->get_dof_transformation_function<T>(true, true);

    // Get interpolation operator
    const auto [_Pi, pi_shape] = element->interpolation_operator();
    cmdspan2_t Pi(_Pi.data(), pi_shape);

    namespace stdex = std::experimental;
    using u_t = stdex::mdspan<const T, stdex::dextents<std::size_t, 2>>;
    using U_t = stdex::mdspan<T, stdex::dextents<std::size_t, 2>>;
    using J_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
    using K_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
    auto pull_back_fn = element->basix_element().map_fn<U_t, u_t, J_t, K_t>();

    for (std::size_t c = 0; c < cells.size(); ++c)
    {
      const std::int32_t cell = cells[c];
      auto x_dofs = x_dofmap.links(cell);
      for (int i = 0; i < num_dofs_g; ++i)
      {
        const int pos = 3 * x_dofs[i];
        for (int j = 0; j < gdim; ++j)
          coord_dofs(i, j) = x_g[pos + j];
      }

      // Compute J, detJ and K
      std::fill(J_b.begin(), J_b.end(), 0);
      for (std::size_t p = 0; p < Xshape[0]; ++p)
      {
        auto _dphi
            = stdex::submdspan(dphi, stdex::full_extent, p, stdex::full_extent);
        auto _J
            = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
        cmap.compute_jacobian(_dphi, coord_dofs, _J);
        auto _K
            = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);
        cmap.compute_jacobian_inverse(_J, _K);
        detJ[p] = cmap.compute_jacobian_determinant(_J, det_scratch);
      }

      std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
      for (int k = 0; k < element_bs; ++k)
      {
        // Extract computed expression values for element block k
        for (int m = 0; m < value_size; ++m)
        {
          for (std::size_t k0 = 0; k0 < Xshape[0]; ++k0)
          {
            _vals(k0, 0, m)
                = f[f_shape1 * (k * value_size + m) + c * Xshape[0] + k0];
          }
        }

        // Get element degrees of freedom for block
        for (std::size_t i = 0; i < Xshape[0]; ++i)
        {
          auto _u = stdex::submdspan(_vals, i, stdex::full_extent,
                                     stdex::full_extent);
          auto _U = stdex::submdspan(ref_data, i, stdex::full_extent,
                                     stdex::full_extent);
          auto _K
              = stdex::submdspan(K, i, stdex::full_extent, stdex::full_extent);
          auto _J
              = stdex::submdspan(J, i, stdex::full_extent, stdex::full_extent);
          pull_back_fn(_U, _u, _K, 1.0 / detJ[i], _J);
        }

        auto ref = stdex::submdspan(ref_data, stdex::full_extent, 0,
                                    stdex::full_extent);
        impl::interpolation_apply(Pi, ref, std::span(_coeffs), element_bs);
        apply_inverse_transpose_dof_transformation(_coeffs, cell_info, cell, 1);

        // Copy interpolation dofs into coefficient vector
        assert(_coeffs.size() == num_scalar_dofs);
        for (int i = 0; i < num_scalar_dofs; ++i)
        {
          const int dof = i * element_bs + k;
          std::div_t pos = std::div(dof, dofmap_bs);
          coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
        }
      }
    }
  }
}

/// Interpolate from one finite element Function to another on the same
/// mesh
/// @param[out] u The function to interpolate into
/// @param[in] v The function to be interpolated
/// @param[in] cells List of cell indices to interpolate on
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v,
                 const std::span<const std::int32_t>& cells)
{
  assert(u.function_space());
  assert(v.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);

  auto cell_map0 = mesh->topology().index_map(mesh->topology().dim());
  assert(cell_map0);
  std::size_t num_cells0 = cell_map0->size_local() + cell_map0->num_ghosts();
  if (u.function_space() == v.function_space() and cells.size() == num_cells0)
  {
    // Same function spaces and on whole mesh
    std::span<T> u1_array = u.x()->mutable_array();
    std::span<const T> u0_array = v.x()->array();
    std::copy(u0_array.begin(), u0_array.end(), u1_array.begin());
  }
  else
  {
    // Get mesh and check that functions share the same mesh
    if (mesh != v.function_space()->mesh())
    {
      throw std::runtime_error(
          "Interpolation on different meshes not supported (yet).");
    }

    // Get elements and check value shape
    auto element0 = v.function_space()->element();
    assert(element0);
    auto element1 = u.function_space()->element();
    assert(element1);
    if (element0->value_shape().size() != element1->value_shape().size()
        or !std::equal(element0->value_shape().begin(),
                       element0->value_shape().end(),
                       element1->value_shape().begin()))
    {
      throw std::runtime_error(
          "Interpolation: elements have different value dimensions");
    }

    if (*element1 == *element0)
    {
      // Same element, different dofmaps (or just a subset of cells)

      const int tdim = mesh->topology().dim();
      auto cell_map = mesh->topology().index_map(tdim);
      assert(cell_map);

      assert(element1->block_size() == element0->block_size());

      // Get dofmaps
      std::shared_ptr<const DofMap> dofmap0 = v.function_space()->dofmap();
      assert(dofmap0);
      std::shared_ptr<const DofMap> dofmap1 = u.function_space()->dofmap();
      assert(dofmap1);

      std::span<T> u1_array = u.x()->mutable_array();
      std::span<const T> u0_array = v.x()->array();

      // Iterate over mesh and interpolate on each cell
      const int bs0 = dofmap0->bs();
      const int bs1 = dofmap1->bs();
      for (auto c : cells)
      {
        std::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(c);
        std::span<const std::int32_t> dofs1 = dofmap1->cell_dofs(c);
        assert(bs0 * dofs0.size() == bs1 * dofs1.size());
        for (std::size_t i = 0; i < dofs0.size(); ++i)
        {
          for (int k = 0; k < bs0; ++k)
          {
            int index = bs0 * i + k;
            std::div_t dv1 = std::div(index, bs1);
            u1_array[bs1 * dofs1[dv1.quot] + dv1.rem]
                = u0_array[bs0 * dofs0[i] + k];
          }
        }
      }
    }
    else if (element1->map_type() == element0->map_type())
    {
      // Different elements, same basis function map type
      impl::interpolate_same_map(u, v, cells);
    }
    else
    {
      //  Different elements with different maps for basis functions
      impl::interpolate_nonmatching_maps(u, v, cells);
    }
  }
}

} // namespace dolfinx::fem
