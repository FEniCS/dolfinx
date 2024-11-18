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
#include "FunctionSpace.h"
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

namespace dolfinx::fem::impl
{
/// @cond
using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const std::int32_t,
    MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
/// @endcond

/// @brief Apply boundary condition lifting for cell integrals.
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
/// @param[in] cells Cell indices (in the integration domain mesh) to
/// execute the kernel over. These are the indices into the geometry
/// dofmap.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param[in] P0 Function that applies transformation P_0 A in-place to
/// transform test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P1T Function that applies transformation A P_1^T in-place
/// to transform trial degrees-of-freedom.
/// @param[in] constants Constants data.
/// @param[in] coeffs The coefficient data array with shape
/// `(cells.size(), cstride)` flattened into row-major format.
/// @param[in] cstride The coefficient stride.
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 The cell permutation information for the trial
/// function mesh.
/// @param[in] bc_values1 The value for entries with an applied boundary
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
    std::span<const T> coeffs, int cstride,
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
    auto dofs1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dmap1, c1, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

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
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Size data structure for assembly
    auto dofs0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dmap0, c0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    const int num_rows = bs0 * dofs0.size();
    const int num_cols = bs1 * dofs1.size();

    const T* coeff_array = coeffs.data() + index * cstride;
    Ae.resize(num_rows * num_cols);
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), coeff_array, constants.data(), coordinate_dofs.data(),
           nullptr, nullptr);
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
/// @tparam T The scalar type
/// @tparam _bs FIXME This is unused
/// @param[in,out] b The vector to modify
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] num_facets_per_cell Number of cell facets
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] facets Facet indices (in the integration domain mesh) to
/// execute the kernel over.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param[in] P0 Function that applies transformation P_0 A in-place to
/// transform test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P1T Function that applies transformation A P_1^T in-place
/// to transform trial degrees-of-freedom.
/// @param[in] constants The constant data.
/// @param[in] coeffs The coefficient data array of shape (cells.size(),
/// cstride), flattened into row-major format.
/// @param[in] cstride The coefficient stride.
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 The cell permutation information for the trial
/// function mesh.
/// @param[in] bc_values1 The value for entries with an applied boundary
/// condition.
/// @param[in] bc_markers1 Marker to identify which DOFs have boundary
/// conditions applied.
/// @param[in] x0 The vector used in the lifting.
/// @param[in] alpha The scaling to apply.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
template <dolfinx::scalar T, int _bs = -1>
void _lift_bc_exterior_facets(
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, int num_facets_per_cell,
    FEkernel<T> auto kernel, std::span<const std::int32_t> facets,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1, std::span<const T> bc_values1,
    std::span<const std::int8_t> bc_markers1, std::span<const T> x0, T alpha,
    std::span<const std::uint8_t> perms)
{
  if (facets.empty())
    return;

  const auto [dmap0, bs0, facets0] = dofmap0;
  const auto [dmap1, bs1, facets1] = dofmap1;

  // Data structures used in bc application
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> Ae, be;
  assert(facets.size() % 2 == 0);
  assert(facets0.size() == facets.size());
  assert(facets1.size() == facets.size());
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    // Cell in integration domain mesh
    std::int32_t cell = facets[index];
    // Cell in test function mesh
    std::int32_t cell0 = facets0[index];
    // Cell in trial function mesh
    std::int32_t cell1 = facets1[index];

    // Local facet index
    std::int32_t local_facet = facets[index + 1];

    // Get dof maps for cell
    auto dofs1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dmap1, cell1, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

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
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Size data structure for assembly
    auto dofs0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dmap0, cell0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    const int num_rows = bs0 * dofs0.size();
    const int num_cols = bs1 * dofs1.size();

    // Permutations
    std::uint8_t perm
        = perms.empty() ? 0 : perms[cell * num_facets_per_cell + local_facet];

    const T* coeff_array = coeffs.data() + index / 2 * cstride;
    Ae.resize(num_rows * num_cols);
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), coeff_array, constants.data(), coordinate_dofs.data(),
           &local_facet, &perm);
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
/// @tparam T Scalar type.
/// @tparam _bs FIXME This is unused.
/// @param[in,out] b The vector to modify
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] num_facets_per_cell Number of facets of a cell.
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] facets Facet indices (in the integration domain mesh) to
/// execute the kernel over.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P0 Function that applies transformation P_0 A in-place to
/// transform test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P1T Function that applies transformation A P_1^T in-place
/// to transform trial degrees-of-freedom.
/// @param[in] constants The constant data.
/// @param[in] coeffs The coefficient data array of shape (cells.size(),
/// cstride), flattened into row-major format.
/// @param[in] cstride The coefficient stride.
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 The cell permutation information for the trial
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
/// @param[in] bc_values1 The value for entries with an applied boundary
/// condition.
/// @param[in] bc_markers1 Marker to identify which DOFs have boundary
/// conditions applied.
/// @param[in] x0 The vector used in the lifting.
/// @param[in] alpha The scaling to apply
template <dolfinx::scalar T, int _bs = -1>
void _lift_bc_interior_facets(
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, int num_facets_per_cell,
    FEkernel<T> auto kernel, std::span<const std::int32_t> facets,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1,
    std::span<const std::uint8_t> perms, std::span<const T> bc_values1,
    std::span<const std::int8_t> bc_markers1, std::span<const T> x0, T alpha)
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
  assert(facets.size() % 4 == 0);

  const int num_dofs0 = dmap0.extent(1);
  const int num_dofs1 = dmap1.extent(1);
  assert(facets0.size() == facets.size());
  assert(facets1.size() == facets.size());
  for (std::size_t index = 0; index < facets.size(); index += 4)
  {
    // Cells in integration domain, test function domain and trial
    // function domain meshes
    std::array cells{facets[index], facets[index + 2]};
    std::array cells0{facets0[index], facets0[index + 2]};
    std::array cells1{facets1[index], facets1[index + 2]};

    // Local facet indices
    std::array<std::int32_t, 2> local_facet
        = {facets[index + 1], facets[index + 3]};

    // Get cell geometry
    auto x_dofs0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cells[0], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cells[1], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    // Get dof maps for cells and pack
    auto dmap0_cell0
        = std::span(dmap0.data_handle() + cells0[0] * num_dofs0, num_dofs0);
    auto dmap0_cell1
        = std::span(dmap0.data_handle() + cells0[1] * num_dofs0, num_dofs0);

    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::ranges::copy(dmap0_cell0, dmapjoint0.begin());
    std::ranges::copy(dmap0_cell1,
                      std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    auto dmap1_cell0
        = std::span(dmap1.data_handle() + cells1[0] * num_dofs1, num_dofs1);
    auto dmap1_cell1
        = std::span(dmap1.data_handle() + cells1[1] * num_dofs1, num_dofs1);

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
    std::array perm
        = perms.empty()
              ? std::array<std::uint8_t, 2>{0, 0}
              : std::array{
                    perms[cells[0] * num_facets_per_cell + local_facet[0]],
                    perms[cells[1] * num_facets_per_cell + local_facet[1]]};
    kernel(Ae.data(), coeffs.data() + index / 2 * cstride, constants.data(),
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

/// @brief Execute kernel over cells and accumulate result in vector
/// @tparam T  The scalar type
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @param P0 Function that applies transformation P0.b in-place to
/// transform test degrees-of-freedom.
/// @param b The vector to accumulate into
/// @param x_dofmap Dofmap for the mesh geometry.
/// @param x Mesh geometry (coordinates).
/// @param cells Cell indices (in the integration domain mesh) to execute
/// the kernel over. These are the indices into the geometry dofmap.
/// @param dofmap Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param kernel Kernel function to execute over each cell.
/// @param constants The constant data
/// @param coeffs The coefficient data array of shape (cells.size(), cstride),
/// flattened into row-major format.
/// @param cstride The coefficient stride
/// @param cell_info0 The cell permutation information for the test function
/// mesh
template <dolfinx::scalar T, int _bs = -1>
void assemble_cells(
    fem::DofTransformKernel<T> auto P0, std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    std::span<const std::int32_t> cells,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap,
    FEkernel<T> auto kernel, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
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
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate vector for cell
    std::ranges::fill(be, 0);
    kernel(be.data(), coeffs.data() + index * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    P0(_be, cell_info0, c0, 1);

    // Scatter cell vector to 'global' vector array
    auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dmap, c0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
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
/// @tparam T The scalar type
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @param P0 Function that applies transformation P0.b in-place to
/// transform test degrees-of-freedom.
/// @param[in,out] b The vector to accumulate into.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] num_facets_per_cell Number of cell facets
/// @param[in] facets Facets (in the integration domain mesh) to execute
/// the kernel over.
/// @param[in] dofmap Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param[in] fn Kernel function to execute over each cell.
/// @param[in] constants The constant data.
/// @param[in] coeffs The coefficient data array of shape
/// `(cells.size(), cstride)`, flattened into row-major format.
/// @param[in] cstride The coefficient stride.
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
template <dolfinx::scalar T, int _bs = -1>
void assemble_exterior_facets(
    fem::DofTransformKernel<T> auto P0, std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, int num_facets_per_cell,
    std::span<const std::int32_t> facets,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap,
    FEkernel<T> auto fn, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint8_t> perms)
{
  if (facets.empty())
    return;

  const auto [dmap, bs, facets0] = dofmap;
  assert(_bs < 0 or _bs == bs);

  // FIXME: Add proper interface for num_dofs
  // Create data structures used in assembly
  const int num_dofs = dmap.extent(1);
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> be(bs * num_dofs);
  std::span<T> _be(be);
  assert(facets.size() % 2 == 0);
  assert(facets0.size() == facets.size());
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    // Cell in the integration domain, local facet index relative to the
    // integration domain cell, and cell in the test function mesh
    std::int32_t cell = facets[index];
    std::int32_t local_facet = facets[index + 1];
    std::int32_t cell0 = facets0[index];

    // Get cell coordinates/geometry
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Permutations
    std::uint8_t perm
        = perms.empty() ? 0 : perms[cell * num_facets_per_cell + local_facet];

    // Tabulate element vector
    std::ranges::fill(be, 0);
    fn(be.data(), coeffs.data() + index / 2 * cstride, constants.data(),
       coordinate_dofs.data(), &local_facet, &perm);

    P0(_be, cell_info0, cell0, 1);

    // Add element vector to global vector
    auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dmap, cell0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
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
/// @param[in] num_facets_per_cell Number of facets of a cell
/// @param[in] facets Facets (in the integration domain mesh) to execute
/// the kernel over.
/// @param[in] dofmap Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param[in] fn Kernel function to execute over each cell.
/// @param[in] constants The constant data
/// @param[in] coeffs The coefficient data array of shape (cells.size(),
/// cstride), flattened into row-major format.
/// @param[in] cstride The coefficient stride
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
template <dolfinx::scalar T, int _bs = -1>
void assemble_interior_facets(
    fem::DofTransformKernel<T> auto P0, std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, int num_facets_per_cell,
    std::span<const std::int32_t> facets,
    std::tuple<const DofMap&, int, std::span<const std::int32_t>> dofmap,
    FEkernel<T> auto fn, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint8_t> perms)
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

  assert(facets.size() % 4 == 0);
  assert(facets0.size() == facets.size());
  for (std::size_t index = 0; index < facets.size(); index += 4)
  {
    // Cells in integration domain and test function domain meshes
    std::array<std::int32_t, 2> cells{facets[index], facets[index + 2]};
    std::array<std::int32_t, 2> cells0{facets0[index], facets0[index + 2]};

    // Local facet indices
    std::array<std::int32_t, 2> local_facet{facets[index + 1],
                                            facets[index + 3]};

    // Get cell geometry
    auto x_dofs0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cells[0], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cells[1], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    // Get dofmaps for cells
    std::span<const std::int32_t> dmap0 = dmap.cell_dofs(cells0[0]);
    std::span<const std::int32_t> dmap1 = dmap.cell_dofs(cells0[1]);

    // Tabulate element vector
    be.resize(bs * (dmap0.size() + dmap1.size()));
    std::ranges::fill(be, 0);
    std::array perm
        = perms.empty()
              ? std::array<std::uint8_t, 2>{0, 0}
              : std::array{
                    perms[cells[0] * num_facets_per_cell + local_facet[0]],
                    perms[cells[1] * num_facets_per_cell + local_facet[1]]};
    fn(be.data(), coeffs.data() + index / 2 * cstride, constants.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data());

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
    auto kernel = a.kernel(IntegralType::cell, i);
    assert(kernel);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = a.domain(IntegralType::cell, i);
    if (bs0 == 1 and bs1 == 1)
    {
      _lift_bc_cells<T, 1, 1>(
          b, x_dofmap, x, kernel, cells,
          {dofmap0, bs0, a.domain(IntegralType::cell, i, *mesh0)}, P0,
          {dofmap1, bs1, a.domain(IntegralType::cell, i, *mesh1)}, P1T,
          constants, coeffs, cstride, cell_info0, cell_info1, bc_values1,
          bc_markers1, x0, alpha);
    }
    else if (bs0 == 3 and bs1 == 3)
    {
      _lift_bc_cells<T, 3, 3>(
          b, x_dofmap, x, kernel, cells,
          {dofmap0, bs0, a.domain(IntegralType::cell, i, *mesh0)}, P0,
          {dofmap1, bs1, a.domain(IntegralType::cell, i, *mesh1)}, P1T,
          constants, coeffs, cstride, cell_info0, cell_info1, bc_values1,
          bc_markers1, x0, alpha);
    }
    else
    {
      _lift_bc_cells(b, x_dofmap, x, kernel, cells,
                     {dofmap0, bs0, a.domain(IntegralType::cell, i, *mesh0)},
                     P0,
                     {dofmap1, bs1, a.domain(IntegralType::cell, i, *mesh1)},
                     P1T, constants, coeffs, cstride, cell_info0, cell_info1,
                     bc_values1, bc_markers1, x0, alpha);
    }
  }

  std::span<const std::uint8_t> perms;
  if (a.needs_facet_permutations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    perms = std::span(mesh->topology()->get_facet_permutations());
  }

  mesh::CellType cell_type = mesh->topology()->cell_type();
  int num_facets_per_cell
      = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
  for (int i : a.integral_ids(IntegralType::exterior_facet))
  {
    auto kernel = a.kernel(IntegralType::exterior_facet, i);
    assert(kernel);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});
    _lift_bc_exterior_facets(
        b, x_dofmap, x, num_facets_per_cell, kernel,
        a.domain(IntegralType::exterior_facet, i),
        {dofmap0, bs0, a.domain(IntegralType::exterior_facet, i, *mesh0)}, P0,
        {dofmap1, bs1, a.domain(IntegralType::exterior_facet, i, *mesh1)}, P1T,
        constants, coeffs, cstride, cell_info0, cell_info1, bc_values1,
        bc_markers1, x0, alpha, perms);
  }

  for (int i : a.integral_ids(IntegralType::interior_facet))
  {
    auto kernel = a.kernel(IntegralType::interior_facet, i);
    assert(kernel);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::interior_facet, i});
    _lift_bc_interior_facets(
        b, x_dofmap, x, num_facets_per_cell, kernel,
        a.domain(IntegralType::interior_facet, i),
        {dofmap0, bs0, a.domain(IntegralType::interior_facet, i, *mesh0)}, P0,
        {dofmap1, bs1, a.domain(IntegralType::interior_facet, i, *mesh1)}, P1T,
        constants, coeffs, cstride, cell_info0, cell_info1, perms, bc_values1,
        bc_markers1, x0, alpha);
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
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where a[j] is the form that
/// generates A_j
/// @param[in] constants Constants that appear in `a`
/// @param[in] coeffs Coefficients that appear in `a`
/// @param[in] bcs1 List of boundary conditions for each block, i.e.
/// bcs1[2] are the boundary conditions applied to the columns of a[2] /
/// x0[2] block
/// @param[in] x0 The vectors used in the lifting
/// @param[in] alpha Scaling to apply
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
/// @param[in,out] b The vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L Linear forms to assemble into b.
/// @param[in] x_dofmap Mesh geometry dofmap.
/// @param[in] x Mesh coordinates.
/// @param[in] constants Packed constants that appear in `L`.
/// @param[in] coefficients Packed coefficients that appear in `L`.
template <dolfinx::scalar T, std::floating_point U>
void assemble_vector(
    std::span<T> b, const Form<T, U>& L, mdspan2_t x_dofmap,
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

  // Get dofmap data
  assert(L.function_spaces().at(0));
  auto element = L.function_spaces().at(0)->element();
  assert(element);
  std::shared_ptr<const fem::DofMap> dofmap
      = L.function_spaces().at(0)->dofmap();
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
    auto fn = L.kernel(IntegralType::cell, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = L.domain(IntegralType::cell, i);
    if (bs == 1)
    {
      impl::assemble_cells<T, 1>(
          P0, b, x_dofmap, x, cells,
          {dofs, bs, L.domain(IntegralType::cell, i, *mesh0)}, fn, constants,
          coeffs, cstride, cell_info0);
    }
    else if (bs == 3)
    {
      impl::assemble_cells<T, 3>(
          P0, b, x_dofmap, x, cells,
          {dofs, bs, L.domain(IntegralType::cell, i, *mesh0)}, fn, constants,
          coeffs, cstride, cell_info0);
    }
    else
    {
      impl::assemble_cells(P0, b, x_dofmap, x, cells,
                           {dofs, bs, L.domain(IntegralType::cell, i, *mesh0)},
                           fn, constants, coeffs, cstride, cell_info0);
    }
  }

  std::span<const std::uint8_t> perms;
  if (L.needs_facet_permutations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    perms = std::span(mesh->topology()->get_facet_permutations());
  }

  mesh::CellType cell_type = mesh->topology()->cell_type();
  int num_facets_per_cell
      = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
  for (int i : L.integral_ids(IntegralType::exterior_facet))
  {
    auto fn = L.kernel(IntegralType::exterior_facet, i);
    assert(fn);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});
    std::span<const std::int32_t> facets
        = L.domain(IntegralType::exterior_facet, i);
    if (bs == 1)
    {
      impl::assemble_exterior_facets<T, 1>(
          P0, b, x_dofmap, x, num_facets_per_cell, facets,
          {dofs, bs, L.domain(IntegralType::exterior_facet, i, *mesh0)}, fn,
          constants, coeffs, cstride, cell_info0, perms);
    }
    else if (bs == 3)
    {
      impl::assemble_exterior_facets<T, 3>(
          P0, b, x_dofmap, x, num_facets_per_cell, facets,
          {dofs, bs, L.domain(IntegralType::exterior_facet, i, *mesh0)}, fn,
          constants, coeffs, cstride, cell_info0, perms);
    }
    else
    {
      impl::assemble_exterior_facets(
          P0, b, x_dofmap, x, num_facets_per_cell, facets,
          {dofs, bs, L.domain(IntegralType::exterior_facet, i, *mesh0)}, fn,
          constants, coeffs, cstride, cell_info0, perms);
    }
  }

  for (int i : L.integral_ids(IntegralType::interior_facet))
  {
    auto fn = L.kernel(IntegralType::interior_facet, i);
    assert(fn);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::interior_facet, i});
    std::span<const std::int32_t> facets
        = L.domain(IntegralType::interior_facet, i);
    if (bs == 1)
    {
      impl::assemble_interior_facets<T, 1>(
          P0, b, x_dofmap, x, num_facets_per_cell, facets,
          {*dofmap, bs, L.domain(IntegralType::interior_facet, i, *mesh0)}, fn,
          constants, coeffs, cstride, cell_info0, perms);
    }
    else if (bs == 3)
    {
      impl::assemble_interior_facets<T, 3>(
          P0, b, x_dofmap, x, num_facets_per_cell, facets,
          {*dofmap, bs, L.domain(IntegralType::interior_facet, i, *mesh0)}, fn,
          constants, coeffs, cstride, cell_info0, perms);
    }
    else
    {
      impl::assemble_interior_facets(
          P0, b, x_dofmap, x, num_facets_per_cell, facets,
          {*dofmap, bs, L.domain(IntegralType::interior_facet, i, *mesh0)}, fn,
          constants, coeffs, cstride, cell_info0, perms);
    }
  }
}

/// @brief Assemble linear form into a vector
/// @param[in,out] b The vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] constants Packed constants that appear in `L`
/// @param[in] coefficients Packed coefficients that appear in `L`
template <dolfinx::scalar T, std::floating_point U>
void assemble_vector(
    std::span<T> b, const Form<T, U>& L, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh<U>> mesh = L.mesh();
  assert(mesh);
  if constexpr (std::is_same_v<U, scalar_value_type_t<T>>)
    assemble_vector(b, L, mesh->geometry().dofmap(), mesh->geometry().x(),
                    constants, coefficients);
  else
  {
    auto x = mesh->geometry().x();
    std::vector<scalar_value_type_t<T>> _x(x.begin(), x.end());
    assemble_vector(b, L, mesh->geometry().dofmap(), _x, constants,
                    coefficients);
  }
}
} // namespace dolfinx::fem::impl
