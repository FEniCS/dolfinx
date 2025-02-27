// Copyright (C) 2018-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "DirichletBC.h"
#include "DofMap.h"
#include "Form.h"
#include "traits.h"
#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace dolfinx::fem
{
template <dolfinx::scalar T, std::floating_point U>
class DirichletBC;

}
namespace dolfinx::fem::impl
{
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

/// @cond
using mdspan2_t = md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>;
/// @endcond

/// @brief Apply boundary condition lifting for cell integrals.
///
/// @tparam T The scalar type.
/// @tparam _bs0 The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs0` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @tparam _bs1 The block size of the trial function dof map.
/// @param[in,out] b Vector to modify.
/// @param x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] cells Cell indices to execute the kernel over. These are
/// the indices into the geometry dofmap `x_dofmap`.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P0 Function that applies transformation `P_0 A` in-place
/// to the computed tensor `A` to transform its test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P1T Function that applies transformation `A P_1^T`
/// in-place to to the computed tensor `A` to transform trial
/// degrees-of-freedom.
/// @param[in] constants Constant data in the kernel.
/// @param[in] coeffs Coefficient data in the kernel. It has shape
/// `(cells.size(), num_cell_coeffs)`. `coeffs(i, j)` is the `j`th
/// coefficient for cell `i`.
/// @param[in] cell_info0 Cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 Cell permutation information for the trial
/// function mesh.
/// @param[in] bc_values1 Value for entries with an applied boundary
/// condition.
/// @param[in] bc_markers1 Marker to identify which DOFs have boundary
/// conditions applied.
/// @param[in] x0 Vector used in the lifting.
/// @param[in] alpha Scaling to apply.
template <dolfinx::scalar T, int _bs0 = -1, int _bs1 = -1>
void _lift_bc_cells(
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, FEkernel<T> auto kernel,
    std::span<const std::int32_t> cells,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1, std::span<const T> bc_values1,
    std::span<const std::int8_t> bc_markers1, std::span<const T> x0, T alpha)
{
  if (cells.empty())
    return;

  const auto [dmap0, bs0, cells0] = dofmap0;
  const auto [dmap1, bs1, cells1] = dofmap1;
  assert(_bs0 < 0 or _bs0 == bs0);
  assert(_bs1 < 0 or _bs1 == bs1);

  // Data structures used in bc application
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> Ae, be;
  assert(cells0.size() == cells.size());
  assert(cells1.size() == cells.size());
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Cell index in integration domain mesh, test function mesh, and trial
    // function mesh
    std::int32_t c = cells[index];
    std::int32_t c0 = cells0[index];
    std::int32_t c1 = cells1[index];

    // Get dof maps for cell
    auto dofs1 = md::submdspan(dmap1, c1, md::full_extent);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (std::size_t j = 0; j < dofs1.size(); ++j)
    {
      if constexpr (_bs1 > 0)
      {
        for (int k = 0; k < _bs1; ++k)
        {
          assert(_bs1 * dofs1[j] + k < (int)bc_markers1.size());
          if (bc_markers1[_bs1 * dofs1[j] + k])
          {
            has_bc = true;
            break;
          }
        }
      }
      else
      {
        for (int k = 0; k < bs1; ++k)
        {
          assert(bs1 * dofs1[j] + k < (int)bc_markers1.size());
          if (bc_markers1[bs1 * dofs1[j] + k])
          {
            has_bc = true;
            break;
          }
        }
      }
    }

    if (!has_bc)
      continue;

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Size data structure for assembly
    auto dofs0 = md::submdspan(dmap0, c0, md::full_extent);

    const int num_rows = bs0 * dofs0.size();
    const int num_cols = bs1 * dofs1.size();

    Ae.resize(num_rows * num_cols);
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), &coeffs(index, 0), constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    P0(Ae, cell_info0, c0, num_cols);
    P1T(Ae, cell_info1, c1, num_rows);

    // Size data structure for assembly
    be.resize(num_rows);
    std::ranges::fill(be, 0);
    for (std::size_t j = 0; j < dofs1.size(); ++j)
    {
      if constexpr (_bs1 > 0)
      {
        for (int k = 0; k < _bs1; ++k)
        {
          const std::int32_t jj = _bs1 * dofs1[j] + k;
          assert(jj < (int)bc_markers1.size());
          if (bc_markers1[jj])
          {
            const T bc = bc_values1[jj];
            const T _x0 = x0.empty() ? 0 : x0[jj];
            // const T _x0 = 0;
            // be -= Ae.col(bs1 * j + k) * alpha * (bc - _x0);
            for (int m = 0; m < num_rows; ++m)
              be[m] -= Ae[m * num_cols + _bs1 * j + k] * alpha * (bc - _x0);
          }
        }
      }
      else
      {
        for (int k = 0; k < bs1; ++k)
        {
          const std::int32_t jj = bs1 * dofs1[j] + k;
          assert(jj < (int)bc_markers1.size());
          if (bc_markers1[jj])
          {
            const T bc = bc_values1[jj];
            const T _x0 = x0.empty() ? 0 : x0[jj];
            // be -= Ae.col(bs1 * j + k) * alpha * (bc - _x0);
            for (int m = 0; m < num_rows; ++m)
              be[m] -= Ae[m * num_cols + bs1 * j + k] * alpha * (bc - _x0);
          }
        }
      }
    }

    for (std::size_t i = 0; i < dofs0.size(); ++i)
    {
      if constexpr (_bs0 > 0)
      {
        for (int k = 0; k < _bs0; ++k)
          b[_bs0 * dofs0[i] + k] += be[_bs0 * i + k];
      }
      else
      {
        for (int k = 0; k < bs0; ++k)
          b[bs0 * dofs0[i] + k] += be[bs0 * i + k];
      }
    }
  }
}

/// @brief Apply lifting for exterior facet integrals.
///
/// @tparam T Scalar type.
/// @param[in,out] b Vector to modify.
/// @param[in] x_dofmap Degree-of-freedom map for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] kernel Kernel function to execute over each facet.
/// @param[in] facets Facets to execute the kernel over, where for the
/// `i`th facet `facets(i, 0)` is the attached cell and `facets(i, 1)`
/// is the local index of the facet relative to the cell.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap
/// indices. See `facets` documentation for the dofmap indices layout.
/// @param[in] P0 Function that applies the transformation `P_0 A`
/// in-place to `A` to transform the test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data.
/// See `dofmap0` for a description.
/// @param[in] P1T Function that applies the transformation `A P_1^T`
/// in-place to `A` to transform the trial degrees-of-freedom.
/// @param[in] constants Constant coefficient data in the kernel.
/// @param[in] coeffs Coefficient data in the kernel. It has shape
/// `(cells.size(), num_cell_coeffs)`. `coeffs(i, j)` is the `j`th
/// coefficient for cell `i`.
/// @param[in] cell_info0 Cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 Cell permutation information for the trial
/// function mesh.
/// @param[in] bc_values1 Values for entries with an applied boundary
/// condition.
/// @param[in] bc_markers1 Marker to identify which DOFs have boundary
/// conditions applied.
/// @param[in] x0 The vector used in the lifting.
/// @param[in] alpha Scaling to apply.
/// @param[in] perms Facet permutation data, where `(cell_idx,
/// local_facet_idx)` is the permutation value for the facet attached to
/// the cell `cell_idx` with local index `local_facet_idx` relative to
/// the cell. Empty if facet permutations are not required.
template <dolfinx::scalar T>
void _lift_bc_exterior_facets(
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, FEkernel<T> auto kernel,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 2>>
        facets,
    std::tuple<mdspan2_t, int,
               md::mdspan<const std::int32_t,
                          md::extents<std::size_t, md::dynamic_extent, 2>>>
        dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int,
               md::mdspan<const std::int32_t,
                          md::extents<std::size_t, md::dynamic_extent, 2>>>
        dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1, std::span<const T> bc_values1,
    std::span<const std::int8_t> bc_markers1, std::span<const T> x0, T alpha,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms)
{
  if (facets.empty())
    return;

  const auto [dmap0, bs0, facets0] = dofmap0;
  const auto [dmap1, bs1, facets1] = dofmap1;

  // Data structures used in bc application
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> Ae, be;
  assert(facets0.size() == facets.size());
  assert(facets1.size() == facets.size());
  for (std::size_t index = 0; index < facets.extent(0); ++index)
  {
    // Cell in integration domain, test function and trial function
    // meshes
    std::int32_t cell = facets(index, 0);
    std::int32_t cell0 = facets0(index, 0);
    std::int32_t cell1 = facets1(index, 0);

    // Local facet index
    std::int32_t local_facet = facets(index, 1);

    // Get dof maps for cell
    auto dofs1 = md::submdspan(dmap1, cell1, md::full_extent);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (std::size_t j = 0; j < dofs1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        if (bc_markers1[bs1 * dofs1[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }

    if (!has_bc)
      continue;

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Size data structure for assembly
    auto dofs0 = md::submdspan(dmap0, cell0, md::full_extent);

    const int num_rows = bs0 * dofs0.size();
    const int num_cols = bs1 * dofs1.size();

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_facet);

    Ae.resize(num_rows * num_cols);
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), &coeffs(index, 0), constants.data(),
           coordinate_dofs.data(), &local_facet, &perm);
    P0(Ae, cell_info0, cell0, num_cols);
    P1T(Ae, cell_info1, cell1, num_rows);

    // Size data structure for assembly
    be.resize(num_rows);
    std::ranges::fill(be, 0);
    for (std::size_t j = 0; j < dofs1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dofs1[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0 : x0[jj];
          // be -= Ae.col(bs1 * j + k) * alpha * (bc - _x0);
          for (int m = 0; m < num_rows; ++m)
            be[m] -= Ae[m * num_cols + bs1 * j + k] * alpha * (bc - _x0);
        }
      }
    }

    for (std::size_t i = 0; i < dofs0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dofs0[i] + k] += be[bs0 * i + k];
  }
}

/// @brief Apply lifting for interior facet integrals.
///
/// @tparam T Scalar type.
/// @param[in,out] b Vector to modify
/// @param[in] x_dofmap Degree-of-freedom map for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] kernel Kernel function to execute over each facet.
/// @param[in] facets Facets to execute the kernel over, where for the
/// `i`th facet `facets(i, 0, 0)` is the first attached cell and
/// `facets(i, 0, 1)` is the local index of the facet relative to the
/// first cell, and `facets(i, 1, 0)` is the second first attached cell
/// and `facets(i, 1, 1)` is the local index of the facet relative to
/// the second cell.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices. See `facets` documentation for the dofmap indices layout.
/// @param[in] P0 Function that applies the transformation `P_0 A`
/// in-place to `A` to transform the test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data.
/// See `dofmap0` for a description.
/// @param[in] P1T Function that applies the transformation `A P_1^T`
/// in-place to `A` to transform the trial degrees-of-freedom.
/// @param[in] constants Constant coefficient data in the kernel.
/// @param[in] coeffs Coefficient data in the kernel. It has shape
/// `(cells.size(), num_cell_coeffs)`. `coeffs(i, j)` is the `j`th
/// coefficient for cell `i`.
/// @param[in] cell_info0 Cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 Cell permutation information for the trial
/// function mesh.
/// @param[in] bc_values1 Values for entries with an applied boundary
/// condition.
/// @param[in] bc_markers1 Marker to identify which DOFs have boundary
/// conditions applied.
/// @param[in] x0 Vector used in the lifting.
/// @param[in] alpha Scaling to apply
/// @param[in] perms Facet permutation data, where `(cell_idx,
/// local_facet_idx)` is the permutation value for the facet attached to
/// the cell `cell_idx` with local index `local_facet_idx` relative to
/// the cell. Empty if facet permutations are not required.
template <dolfinx::scalar T>
void _lift_bc_interior_facets(
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, FEkernel<T> auto kernel,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 2, 2>>
        facets,
    std::tuple<mdspan2_t, int,
               md::mdspan<const std::int32_t,
                          md::extents<std::size_t, md::dynamic_extent, 2, 2>>>
        dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int,
               md::mdspan<const std::int32_t,
                          md::extents<std::size_t, md::dynamic_extent, 2, 2>>>
        dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const T> constants,
    md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                    md::dynamic_extent>>
        coeffs,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1, std::span<const T> bc_values1,
    std::span<const std::int8_t> bc_markers1, std::span<const T> x0, T alpha,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms)
{
  if (facets.empty())
    return;

  const auto [dmap0, bs0, facets0] = dofmap0;
  const auto [dmap1, bs1, facets1] = dofmap1;

  // Data structures used in assembly
  using X = scalar_value_type_t<T>;
  std::vector<X> coordinate_dofs(2 * x_dofmap.extent(1) * 3);
  std::span<X> cdofs0(coordinate_dofs.data(), x_dofmap.extent(1) * 3);
  std::span<X> cdofs1(coordinate_dofs.data() + x_dofmap.extent(1) * 3,
                      x_dofmap.extent(1) * 3);
  std::vector<T> Ae, be;

  // Temporaries for joint dofmaps
  std::vector<std::int32_t> dmapjoint0, dmapjoint1;

  const int num_dofs0 = dmap0.extent(1);
  const int num_dofs1 = dmap1.extent(1);
  assert(facets0.size() == facets.size());
  assert(facets1.size() == facets.size());
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    // Cells in integration domain, test function domain and trial
    // function domain meshes
    std::array cells{facets(f, 0, 0), facets(f, 1, 0)};
    std::array cells0{facets0(f, 0, 0), facets0(f, 1, 0)};
    std::array cells1{facets1(f, 0, 0), facets1(f, 1, 0)};

    // Local facet indices
    std::array local_facet = {facets(f, 0, 1), facets(f, 1, 1)};

    // Get cell geometry
    auto x_dofs0 = md::submdspan(x_dofmap, cells[0], md::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = md::submdspan(x_dofmap, cells[1], md::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    // Get dof maps for cells and pack
    std::span dmap0_cell0(dmap0.data_handle() + cells0[0] * num_dofs0,
                          num_dofs0);
    std::span dmap0_cell1(dmap0.data_handle() + cells0[1] * num_dofs0,
                          num_dofs0);

    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::ranges::copy(dmap0_cell0, dmapjoint0.begin());
    std::ranges::copy(dmap0_cell1,
                      std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    std::span dmap1_cell0(dmap1.data_handle() + cells1[0] * num_dofs1,
                          num_dofs1);
    std::span dmap1_cell1(dmap1.data_handle() + cells1[1] * num_dofs1,
                          num_dofs1);

    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    std::ranges::copy(dmap1_cell0, dmapjoint1.begin());
    std::ranges::copy(dmap1_cell1,
                      std::next(dmapjoint1.begin(), dmap1_cell0.size()));

    // Check if bc is applied to cell0
    bool has_bc = false;
    for (std::size_t j = 0; j < dmap1_cell0.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        if (bc_markers1[bs1 * dmap1_cell0[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }

    // Check if bc is applied to cell1
    for (std::size_t j = 0; j < dmap1_cell1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        if (bc_markers1[bs1 * dmap1_cell1[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }

    if (!has_bc)
      continue;

    const int num_rows = bs0 * dmapjoint0.size();
    const int num_cols = bs1 * dmapjoint1.size();

    // Tabulate tensor
    Ae.resize(num_rows * num_cols);
    std::ranges::fill(Ae, 0);
    std::array perm = perms.empty()
                          ? std::array<std::uint8_t, 2>{0, 0}
                          : std::array{perms(cells[0], local_facet[0]),
                                       perms(cells[1], local_facet[1])};
    kernel(Ae.data(), &coeffs(f, 0, 0), constants.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data());

    std::span<T> _Ae(Ae);
    std::span<T> sub_Ae0 = _Ae.subspan(bs0 * dmap0_cell0.size() * num_cols,
                                       bs0 * dmap0_cell1.size() * num_cols);

    P0(_Ae, cell_info0, cells0[0], num_cols);
    P0(sub_Ae0, cell_info0, cells0[1], num_cols);
    P1T(_Ae, cell_info1, cells1[0], num_rows);

    for (int row = 0; row < num_rows; ++row)
    {
      // DOFs for dmap1 and cell1 are not stored contiguously in
      // the block matrix, so each row needs a separate span access
      std::span<T> sub_Ae1 = _Ae.subspan(
          row * num_cols + bs1 * dmap1_cell0.size(), bs1 * dmap1_cell1.size());
      P1T(sub_Ae1, cell_info1, cells1[1], 1);
    }

    be.resize(num_rows);
    std::ranges::fill(be, 0);

    // Compute b = b - A*b for cell0
    for (std::size_t j = 0; j < dmap1_cell0.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1_cell0[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0 : x0[jj];
          // be -= Ae.col(bs1 * j + k) * alpha * (bc - _x0);
          for (int m = 0; m < num_rows; ++m)
            be[m] -= Ae[m * num_cols + bs1 * j + k] * alpha * (bc - _x0);
        }
      }
    }

    // Compute b = b - A*b for cell1
    const int offset = bs1 * dmap1_cell0.size();
    for (std::size_t j = 0; j < dmap1_cell1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1_cell1[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0 : x0[jj];
          // be -= Ae.col(offset + bs1 * j + k) * alpha * (bc - x0[jj]);
          for (int m = 0; m < num_rows; ++m)
          {
            be[m]
                -= Ae[m * num_cols + offset + bs1 * j + k] * alpha * (bc - _x0);
          }
        }
      }
    }

    for (std::size_t i = 0; i < dmap0_cell0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0_cell0[i] + k] += be[bs0 * i + k];

    const int offset_be = bs0 * dmap0_cell0.size();
    for (std::size_t i = 0; i < dmap0_cell1.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0_cell1[i] + k] += be[offset_be + bs0 * i + k];
  }
}

/// @brief Execute kernel over cells and accumulate result in vector.
///
/// @tparam T  Scalar type
/// @tparam _bs Block size of the form test function dof map. If less
/// than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @param[in] P0 Function that applies transformation `P0.b` in-place
/// to `b` to transform test degrees-of-freedom.
/// @param[in,out] b Aray to accumulate into.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] cells Cell indices to execute the kernel over. These are
/// the indices into the geometry dofmap.
/// @param[in] dofmap Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] constants Constant coefficient data in the kernel.
/// @param[in] coeffs Coefficient data in the kernel. It has shape
/// `(cells.size(), num_cell_coeffs)`. `coeffs(i, j)` is the `j`th
/// coefficient for cell `i`.
/// @param[in] cell_info0 Cell permutation information for the test
/// function mesh.
template <dolfinx::scalar T, int _bs = -1>
void assemble_cells(
    fem::DofTransformKernel<T> auto P0, std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    std::span<const std::int32_t> cells,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap,
    FEkernel<T> auto kernel, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const std::uint32_t> cell_info0)
{
  if (cells.empty())
    return;

  const auto [dmap, bs, cells0] = dofmap;
  assert(_bs < 0 or _bs == bs);

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> be(bs * dmap.extent(1));
  std::span<T> _be(be);

  // Iterate over active cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Integration domain celland test function cell
    std::int32_t c = cells[index];
    std::int32_t c0 = cells0[index];

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate vector for cell
    std::ranges::fill(be, 0);
    kernel(be.data(), &coeffs(index, 0), constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    P0(_be, cell_info0, c0, 1);

    // Scatter cell vector to 'global' vector array
    auto dofs = md::submdspan(dmap, c0, md::full_extent);
    if constexpr (_bs > 0)
    {
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (int k = 0; k < _bs; ++k)
          b[_bs * dofs[i] + k] += be[_bs * i + k];
    }
    else
    {
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (int k = 0; k < bs; ++k)
          b[bs * dofs[i] + k] += be[bs * i + k];
    }
  }
}

/// @brief Execute kernel over cells and accumulate result in vector.
///
/// @tparam T Scalar type.
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @param P0 Function that applies transformation `P0.b` in-place to
/// transform test degrees-of-freedom.
/// @param[in,out] b The vector to accumulate into.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] facets Facets (in the integration domain mesh) to execute
/// the kernel over.
/// @param[in] dofmap Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param[in] fn Kernel function to execute over each cell.
/// @param[in] constants The constant data.
/// @param[in] coeffs The coefficient data array of shape
/// `(cells.size(), coeffs_per_cell)`.
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
template <dolfinx::scalar T, int _bs = -1>
void assemble_exterior_facets(
    fem::DofTransformKernel<T> auto P0, std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2>>
        facets,
    std::tuple<mdspan2_t, int,
               md::mdspan<const std::int32_t,
                          std::extents<std::size_t, md::dynamic_extent, 2>>>
        dofmap,
    FEkernel<T> auto fn, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const std::uint32_t> cell_info0,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms)
{
  if (facets.empty())
    return;

  const auto [dmap, bs, facets0] = dofmap;
  assert(_bs < 0 or _bs == bs);

  // Create data structures used in assembly
  const int num_dofs = dmap.extent(1);
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> be(bs * num_dofs);
  std::span<T> _be(be);
  assert(facets0.size() == facets.size());
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    // Cell in the integration domain, local facet index relative to the
    // integration domain cell, and cell in the test function mesh
    std::int32_t cell = facets(f, 0);
    std::int32_t local_facet = facets(f, 1);
    std::int32_t cell0 = facets0(f, 0);

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_facet);

    // Tabulate element vector
    std::ranges::fill(be, 0);
    fn(be.data(), &coeffs(f, 0), constants.data(), coordinate_dofs.data(),
       &local_facet, &perm);

    P0(_be, cell_info0, cell0, 1);

    // Add element vector to global vector
    auto dofs = md::submdspan(dmap, cell0, md::full_extent);
    if constexpr (_bs > 0)
    {
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (int k = 0; k < _bs; ++k)
          b[_bs * dofs[i] + k] += be[_bs * i + k];
    }
    else
    {
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (int k = 0; k < bs; ++k)
          b[bs * dofs[i] + k] += be[bs * i + k];
    }
  }
}

/// @brief Assemble linear form interior facet integrals into an vector.
/// @tparam T Scalar type.
/// @tparam _bs Block size of the form test function dof map. If less
/// than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @param P0 Function that applies transformation P0.A in-place to
/// transform trial degrees-of-freedom.
/// @param[in,out] b The vector to accumulate into.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] facets Facets (in the integration domain mesh) to execute
/// the kernel over.
/// @param[in] dofmap Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param[in] fn Kernel function to execute over each cell.
/// @param[in] constants The constant data
/// @param[in] coeffs Coefficient data array, withshape (cells.size(),
/// cstride).
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
template <dolfinx::scalar T, int _bs = -1>
void assemble_interior_facets(
    fem::DofTransformKernel<T> auto P0, std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2, 2>>
        facets,
    std::tuple<const DofMap&, int,
               md::mdspan<const std::int32_t,
                          std::extents<std::size_t, md::dynamic_extent, 2, 2>>>
        dofmap,
    FEkernel<T> auto fn, std::span<const T> constants,
    md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                    md::dynamic_extent>>
        coeffs,
    std::span<const std::uint32_t> cell_info0,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms)
{
  if (facets.empty())
    return;

  const auto [dmap, bs, facets0] = dofmap;
  assert(_bs < 0 or _bs == bs);

  // Create data structures used in assembly
  using X = scalar_value_type_t<T>;
  std::vector<X> coordinate_dofs(2 * x_dofmap.extent(1) * 3);
  std::span<X> cdofs0(coordinate_dofs.data(), x_dofmap.extent(1) * 3);
  std::span<X> cdofs1(coordinate_dofs.data() + x_dofmap.extent(1) * 3,
                      x_dofmap.extent(1) * 3);
  std::vector<T> be;

  assert(facets0.size() == facets.size());
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    // Cells in integration domain and test function domain meshes
    std::array<std::int32_t, 2> cells{facets(f, 0, 0), facets(f, 1, 0)};
    std::array<std::int32_t, 2> cells0{facets0(f, 0, 0), facets0(f, 1, 0)};

    // Local facet indices
    std::array<std::int32_t, 2> local_facet{facets(f, 0, 1), facets(f, 1, 1)};

    // Get cell geometry
    auto x_dofs0 = md::submdspan(x_dofmap, cells[0], md::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = md::submdspan(x_dofmap, cells[1], md::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    // Get dofmaps for cells
    std::span dmap0 = dmap.cell_dofs(cells0[0]);
    std::span dmap1 = dmap.cell_dofs(cells0[1]);

    // Tabulate element vector
    be.resize(bs * (dmap0.size() + dmap1.size()));
    std::ranges::fill(be, 0);
    std::array perm = perms.empty()
                          ? std::array<std::uint8_t, 2>{0, 0}
                          : std::array{perms(cells[0], local_facet[0]),
                                       perms(cells[1], local_facet[1])};
    fn(be.data(), &coeffs(f, 0, 0), constants.data(), coordinate_dofs.data(),
       local_facet.data(), perm.data());

    std::span<T> _be(be);
    std::span<T> sub_be = _be.subspan(bs * dmap0.size(), bs * dmap1.size());

    P0(be, cell_info0, cells0[0], 1);
    P0(sub_be, cell_info0, cells0[1], 1);

    // Add element vector to global vector
    if constexpr (_bs > 0)
    {
      for (std::size_t i = 0; i < dmap0.size(); ++i)
        for (int k = 0; k < _bs; ++k)
          b[_bs * dmap0[i] + k] += be[_bs * i + k];
      for (std::size_t i = 0; i < dmap1.size(); ++i)
        for (int k = 0; k < _bs; ++k)
          b[_bs * dmap1[i] + k] += be[_bs * (i + dmap0.size()) + k];
    }
    else
    {
      for (std::size_t i = 0; i < dmap0.size(); ++i)
        for (int k = 0; k < bs; ++k)
          b[bs * dmap0[i] + k] += be[bs * i + k];
      for (std::size_t i = 0; i < dmap1.size(); ++i)
        for (int k = 0; k < bs; ++k)
          b[bs * dmap1[i] + k] += be[bs * (i + dmap0.size()) + k];
    }
  }
}

/// Modify RHS vector to account for boundary condition such that:
///
/// b <- b - alpha * A.(x_bc - x0)
///
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form that generates A
/// @param[in] x_dofmap Mesh geometry dofmap
/// @param[in] x Mesh coordinates
/// @param[in] constants Constants that appear in `a`
/// @param[in] coefficients Coefficients that appear in `a`
/// @param[in] bc_values1 The boundary condition 'values'
/// @param[in] bc_markers1 The indices (columns of A, rows of x) to
/// which bcs belong
/// @param[in] x0 The array used in the lifting, typically a 'current
/// solution' in a Newton method
/// @param[in] alpha Scaling to apply
template <dolfinx::scalar T, std::floating_point U>
void lift_bc(std::span<T> b, const Form<T, U>& a, mdspan2_t x_dofmap,
             std::span<const scalar_value_type_t<T>> x,
             std::span<const T> constants,
             const std::map<std::pair<IntegralType, int>,
                            std::pair<std::span<const T>, int>>& coefficients,
             std::span<const T> bc_values1,
             std::span<const std::int8_t> bc_markers1, std::span<const T> x0,
             T alpha)
{
  // Integration domain mesh
  std::shared_ptr<const mesh::Mesh<U>> mesh = a.mesh();
  assert(mesh);

  // Test function mesh
  auto mesh0 = a.function_spaces().at(0)->mesh();
  assert(mesh0);

  // Trial function mesh
  auto mesh1 = a.function_spaces().at(1)->mesh();
  assert(mesh1);

  // Get dofmap for columns and rows of a
  assert(a.function_spaces().at(0));
  assert(a.function_spaces().at(1));
  auto dofmap0 = a.function_spaces()[0]->dofmap()->map();
  const int bs0 = a.function_spaces()[0]->dofmap()->bs();
  auto element0 = a.function_spaces()[0]->element();
  auto dofmap1 = a.function_spaces()[1]->dofmap()->map();
  const int bs1 = a.function_spaces()[1]->dofmap()->bs();
  auto element1 = a.function_spaces()[1]->element();
  assert(element0);

  std::span<const std::uint32_t> cell_info0;
  std::span<const std::uint32_t> cell_info1;
  // TODO: Check for each element instead
  if (element0->needs_dof_transformations()
      or element1->needs_dof_transformations() or a.needs_facet_permutations())
  {
    mesh0->topology_mutable()->create_entity_permutations();
    mesh1->topology_mutable()->create_entity_permutations();
    cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
  }

  fem::DofTransformKernel<T> auto P0
      = element0->template dof_transformation_fn<T>(doftransform::standard);
  fem::DofTransformKernel<T> auto P1T
      = element1->template dof_transformation_right_fn<T>(
          doftransform::transpose);

  for (int i : a.integral_ids(IntegralType::cell))
  {
    auto kernel = a.kernel(IntegralType::cell, i, 0);
    assert(kernel);
    auto& [_coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span cells = a.domain(IntegralType::cell, i, 0);
    std::span cells0 = a.domain_arg(IntegralType::cell, 0, i, 0);
    std::span cells1 = a.domain_arg(IntegralType::cell, 1, i, 0);
    assert(_coeffs.size() == cells.size() * cstride);
    auto coeffs = md::mdspan(_coeffs.data(), cells.size(), cstride);
    if (bs0 == 1 and bs1 == 1)
    {
      _lift_bc_cells<T, 1, 1>(
          b, x_dofmap, x, kernel, cells, {dofmap0, bs0, cells0}, P0,
          {dofmap1, bs1, cells1}, P1T, constants, coeffs, cell_info0,
          cell_info1, bc_values1, bc_markers1, x0, alpha);
    }
    else if (bs0 == 3 and bs1 == 3)
    {
      _lift_bc_cells<T, 3, 3>(
          b, x_dofmap, x, kernel, cells, {dofmap0, bs0, cells0}, P0,
          {dofmap1, bs1, cells1}, P1T, constants, coeffs, cell_info0,
          cell_info1, bc_values1, bc_markers1, x0, alpha);
    }
    else
    {
      _lift_bc_cells(b, x_dofmap, x, kernel, cells, {dofmap0, bs0, cells0}, P0,
                     {dofmap1, bs1, cells1}, P1T, constants, coeffs, cell_info0,
                     cell_info1, bc_values1, bc_markers1, x0, alpha);
    }
  }

  md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms;
  if (a.needs_facet_permutations())
  {
    mesh::CellType cell_type = mesh->topology()->cell_type();
    int num_facets_per_cell
        = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
    mesh->topology_mutable()->create_entity_permutations();
    const std::vector<std::uint8_t>& p
        = mesh->topology()->get_facet_permutations();
    perms = md::mdspan(p.data(), p.size() / num_facets_per_cell,
                       num_facets_per_cell);
  }

  for (int i : a.integral_ids(IntegralType::exterior_facet))
  {
    auto kernel = a.kernel(IntegralType::exterior_facet, i, 0);
    assert(kernel);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});

    using mdspanx2_t
        = md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>;
    std::span f = a.domain(IntegralType::exterior_facet, i, 0);
    mdspanx2_t facets(f.data(), f.size() / 2, 2);
    std::span f0 = a.domain_arg(IntegralType::exterior_facet, 0, i, 0);
    mdspanx2_t facets0(f0.data(), f0.size() / 2, 2);
    std::span f1 = a.domain_arg(IntegralType::exterior_facet, 1, i, 0);
    mdspanx2_t facets1(f1.data(), f1.size() / 2, 2);
    assert(coeffs.size() == facets.extent(0) * cstride);
    _lift_bc_exterior_facets(
        b, x_dofmap, x, kernel, facets, {dofmap0, bs0, facets0}, P0,
        {dofmap1, bs1, facets1}, P1T, constants,
        md::mdspan(coeffs.data(), facets.extent(0), cstride), cell_info0,
        cell_info1, bc_values1, bc_markers1, x0, alpha, perms);
  }

  for (int i : a.integral_ids(IntegralType::interior_facet))
  {
    auto kernel = a.kernel(IntegralType::interior_facet, i, 0);
    assert(kernel);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::interior_facet, i});

    using mdspanx22_t
        = md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2, 2>>;
    using mdspanx2x_t
        = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                          md::dynamic_extent>>;
    std::span f = a.domain(IntegralType::interior_facet, i, 0);
    mdspanx22_t facets(f.data(), f.size() / 4, 2, 2);
    std::span f0 = a.domain_arg(IntegralType::interior_facet, 0, i, 0);
    mdspanx22_t facets0(f0.data(), f0.size() / 4, 2, 2);
    std::span f1 = a.domain_arg(IntegralType::interior_facet, 1, i, 0);
    mdspanx22_t facets1(f1.data(), f1.size() / 4, 2, 2);
    _lift_bc_interior_facets(
        b, x_dofmap, x, kernel, facets, {dofmap0, bs0, facets0}, P0,
        {dofmap1, bs1, facets1}, P1T, constants,
        mdspanx2x_t(coeffs.data(), facets.extent(0), 2, cstride), cell_info0,
        cell_info1, bc_values1, bc_markers1, x0, alpha, perms);
  }
}

/// Modify b such that:
///
///   b <- b - alpha * A_j.(g_j - x0_j)
///
/// where j is a block (nest) row index. For a non-blocked problem j =
/// 0. The boundary conditions bc1 are on the trial spaces V_j. The
/// forms in [a] must have the same test space as L (from which b was
/// built), but the trial space may differ. If x0 is not supplied, then
/// it is treated as zero.
///
/// @param[in,out] b Array to be modified.
/// @param[in] a Bilinear forms, where `a[j]` is the form that generates
/// `A_j`.
/// @param[in] constants Constants that appear in `a`.
/// @param[in] coeffs Coefficients that appear in `a`.
/// @param[in] bcs1 List of boundary conditions for each block, i.e.
/// `bcs1[2]` are the boundary conditions applied to the columns of
/// `a[2]`/ `x0[2]` block.
/// @param[in] x0 Arrays used in the lifting.
/// @param[in] alpha Scaling to apply.
template <dolfinx::scalar T, std::floating_point U>
void apply_lifting(
    std::span<T> b,
    std::vector<std::optional<std::reference_wrapper<const Form<T, U>>>> a,
    const std::vector<std::span<const T>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const T>, int>>>& coeffs,
    const std::vector<
        std::vector<std::reference_wrapper<const DirichletBC<T, U>>>>& bcs1,
    const std::vector<std::span<const T>>& x0, T alpha)
{
  if (!x0.empty() and x0.size() != a.size())
  {
    throw std::runtime_error(
        "Mismatch in size between x0 and bilinear form in assembler.");
  }

  if (a.size() != bcs1.size())
  {
    throw std::runtime_error(
        "Mismatch in size between a and bcs in assembler.");
  }

  for (std::size_t j = 0; j < a.size(); ++j)
  {
    std::vector<std::int8_t> bc_markers1;
    std::vector<T> bc_values1;
    if (a[j] and !bcs1[j].empty())
    {
      // Extract data from mesh
      std::shared_ptr<const mesh::Mesh<U>> mesh = a[j]->get().mesh();
      if (!mesh)
        throw std::runtime_error("Unable to extract a mesh.");
      mdspan2_t x_dofmap = mesh->geometry().dofmap();
      auto x = mesh->geometry().x();

      assert(a[j]->get().function_spaces().at(0));
      auto V1 = a[j]->get().function_spaces()[1];
      assert(V1);
      auto map1 = V1->dofmap()->index_map;
      const int bs1 = V1->dofmap()->index_map_bs();
      assert(map1);
      const int crange = bs1 * (map1->size_local() + map1->num_ghosts());
      bc_markers1.assign(crange, false);
      bc_values1.assign(crange, 0);
      for (auto& bc : bcs1[j])
      {
        bc.get().mark_dofs(bc_markers1);
        bc.get().set(bc_values1, std::nullopt, 1);
      }

      if (!x0.empty())
      {
        lift_bc<T>(b, a[j]->get(), x_dofmap, x, constants[j], coeffs[j],
                   bc_values1, bc_markers1, x0[j], alpha);
      }
      else
      {
        lift_bc<T>(b, a[j]->get(), x_dofmap, x, constants[j], coeffs[j],
                   bc_values1, bc_markers1, std::span<const T>(), alpha);
      }
    }
  }
}

/// @brief Assemble linear form into a vector.
/// @param[in,out] b Array to be accumulated into. It will not be zeroed
/// before assembly.
/// @param[in] L Linear forms to assemble into b.
/// @param[in] x Mesh coordinates.
/// @param[in] constants Packed constants that appear in `L`.
/// @param[in] coefficients Packed coefficients that appear in `L`.
template <dolfinx::scalar T, std::floating_point U>
void assemble_vector(
    std::span<T> b, const Form<T, U>& L,
    std::span<const scalar_value_type_t<T>> x, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  // Integration domain mesh
  std::shared_ptr<const mesh::Mesh<U>> mesh = L.mesh();
  assert(mesh);

  // Test function mesh
  auto mesh0 = L.function_spaces().at(0)->mesh();
  assert(mesh0);

  const int num_cell_types = mesh->topology()->cell_types().size();
  for (int cell_type_idx = 0; cell_type_idx < num_cell_types; ++cell_type_idx)
  {
    // Geometry dofmap and data
    mdspan2_t x_dofmap = mesh->geometry().dofmap(cell_type_idx);

    // Get dofmap data
    assert(L.function_spaces().at(0));
    auto element = L.function_spaces().at(0)->elements(cell_type_idx);
    assert(element);
    std::shared_ptr<const fem::DofMap> dofmap
        = L.function_spaces().at(0)->dofmaps(cell_type_idx);
    assert(dofmap);
    auto dofs = dofmap->map();
    const int bs = dofmap->bs();

    fem::DofTransformKernel<T> auto P0
        = element->template dof_transformation_fn<T>(doftransform::standard);

    std::span<const std::uint32_t> cell_info0;
    if (element->needs_dof_transformations() or L.needs_facet_permutations())
    {
      mesh0->topology_mutable()->create_entity_permutations();
      cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    }

    for (int i : L.integral_ids(IntegralType::cell))
    {
      auto fn = L.kernel(IntegralType::cell, i, cell_type_idx);
      assert(fn);
      std::span cells = L.domain(IntegralType::cell, i, cell_type_idx);
      std::span cells0 = L.domain_arg(IntegralType::cell, 0, i, cell_type_idx);
      auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
      assert(cells.size() * cstride == coeffs.size());
      if (bs == 1)
      {
        impl::assemble_cells<T, 1>(
            P0, b, x_dofmap, x, cells, {dofs, bs, cells0}, fn, constants,
            md::mdspan(coeffs.data(), cells.size(), cstride), cell_info0);
      }
      else if (bs == 3)
      {
        impl::assemble_cells<T, 3>(
            P0, b, x_dofmap, x, cells, {dofs, bs, cells0}, fn, constants,
            md::mdspan(coeffs.data(), cells.size(), cstride), cell_info0);
      }
      else
      {
        impl::assemble_cells(
            P0, b, x_dofmap, x, cells, {dofs, bs, cells0}, fn, constants,
            md::mdspan(coeffs.data(), cells.size(), cstride), cell_info0);
      }
    }

    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms;
    if (L.needs_facet_permutations())
    {
      mesh::CellType cell_type = mesh->topology()->cell_types()[cell_type_idx];
      int num_facets_per_cell
          = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
      mesh->topology_mutable()->create_entity_permutations();
      const std::vector<std::uint8_t>& p
          = mesh->topology()->get_facet_permutations();
      perms = md::mdspan(p.data(), p.size() / num_facets_per_cell,
                         num_facets_per_cell);
    }

    using mdspanx2_t
        = md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>;

    for (int i : L.integral_ids(IntegralType::exterior_facet))
    {
      auto fn = L.kernel(IntegralType::exterior_facet, i, 0);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::exterior_facet, i});
      std::span f = L.domain(IntegralType::exterior_facet, i, 0);
      mdspanx2_t facets(f.data(), f.size() / 2, 2);
      std::span f1 = L.domain_arg(IntegralType::exterior_facet, 0, i, 0);
      mdspanx2_t facets1(f1.data(), f1.size() / 2, 2);
      assert((facets.size() / 2) * cstride == coeffs.size());
      if (bs == 1)
      {
        impl::assemble_exterior_facets<T, 1>(
            P0, b, x_dofmap, x, facets, {dofs, bs, facets1}, fn, constants,
            md::mdspan(coeffs.data(), facets.extent(0), cstride), cell_info0,
            perms);
      }
      else if (bs == 3)
      {
        impl::assemble_exterior_facets<T, 3>(
            P0, b, x_dofmap, x, facets, {dofs, bs, facets1}, fn, constants,
            md::mdspan(coeffs.data(), facets.size() / 2, cstride), cell_info0,
            perms);
      }
      else
      {
        impl::assemble_exterior_facets(
            P0, b, x_dofmap, x, facets, {dofs, bs, facets1}, fn, constants,
            md::mdspan(coeffs.data(), facets.size() / 2, cstride), cell_info0,
            perms);
      }
    }

    for (int i : L.integral_ids(IntegralType::interior_facet))
    {
      using mdspanx22_t
          = md::mdspan<const std::int32_t,
                       md::extents<std::size_t, md::dynamic_extent, 2, 2>>;
      using mdspanx2x_t
          = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                            md::dynamic_extent>>;

      auto fn = L.kernel(IntegralType::interior_facet, i, 0);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      std::span facets = L.domain(IntegralType::interior_facet, i, 0);
      std::span facets1 = L.domain_arg(IntegralType::interior_facet, 0, i, 0);
      assert((facets.size() / 4) * 2 * cstride == coeffs.size());
      if (bs == 1)
      {
        impl::assemble_interior_facets<T, 1>(
            P0, b, x_dofmap, x,
            mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
            {*dofmap, bs,
             mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
            fn, constants,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            cell_info0, perms);
      }
      else if (bs == 3)
      {
        impl::assemble_interior_facets<T, 3>(
            P0, b, x_dofmap, x,
            mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
            {*dofmap, bs,
             mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
            fn, constants,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            cell_info0, perms);
      }
      else
      {
        impl::assemble_interior_facets(
            P0, b, x_dofmap, x,
            mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
            {*dofmap, bs,
             mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
            fn, constants,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            cell_info0, perms);
      }
    }
  }
}

/// @brief Assemble linear form into a vector.
/// @param[in,out] b Array to accumulate into. It will not be zeroed
/// before assembly.
/// @param[in] L Linear forms to assemble into b.
/// @param[in] constants Packed constants that appear in `L`.
/// @param[in] coefficients Packed coefficients that appear in `L.`
template <dolfinx::scalar T, std::floating_point U>
void assemble_vector(
    std::span<T> b, const Form<T, U>& L, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh<U>> mesh = L.mesh();
  assert(mesh);
  if constexpr (std::is_same_v<U, scalar_value_type_t<T>>)
  {
    assemble_vector(b, L, mesh->geometry().x(), constants, coefficients);
  }
  else
  {
    auto x = mesh->geometry().x();
    std::vector<scalar_value_type_t<T>> _x(x.begin(), x.end());
    assemble_vector(b, L, _x, constants, coefficients);
  }
}
} // namespace dolfinx::fem::impl
