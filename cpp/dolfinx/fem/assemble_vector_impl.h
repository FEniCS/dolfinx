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
#include "utils.h"
#include <algorithm>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace dolfinx::fem::impl
{

namespace stdex = std::experimental;
using mdspan2_t
    = stdex::mdspan<const std::int32_t, stdex::dextents<std::size_t, 2>>;

/// Implementation of vector assembly

/// @brief Implementation of bc application
/// @tparam T The scalar type
/// @tparam _bs0 The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs0` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @tparam _bs1 The block size of the trial function dof map.
template <typename T, int _bs0 = -1, int _bs1 = -1>
void _lift_bc_cells(
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, FEkernel<T> auto kernel,
    std::span<const std::int32_t> cells,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    mdspan2_t dofmap0, int bs0,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    mdspan2_t dofmap1, int bs1, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info, std::span<const T> bc_values1,
    std::span<const std::int8_t> bc_markers1, std::span<const T> x0, T scale)
{
  assert(_bs0 < 0 or _bs0 == bs0);
  assert(_bs1 < 0 or _bs1 == bs1);

  if (cells.empty())
    return;

  // Data structures used in bc application
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> Ae, be;
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];

    // Get dof maps for cell
    auto dmap1 = stdex::submdspan(dofmap1, c, stdex::full_extent);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      if constexpr (_bs1 > 0)
      {
        for (int k = 0; k < _bs1; ++k)
        {
          assert(_bs1 * dmap1[j] + k < (int)bc_markers1.size());
          if (bc_markers1[_bs1 * dmap1[j] + k])
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
          assert(bs1 * dmap1[j] + k < (int)bc_markers1.size());
          if (bc_markers1[bs1 * dmap1[j] + k])
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
    auto x_dofs = stdex::submdspan(x_dofmap, c, stdex::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Size data structure for assembly
    auto dmap0 = stdex::submdspan(dofmap0, c, stdex::full_extent);

    const int num_rows = bs0 * dmap0.size();
    const int num_cols = bs1 * dmap1.size();

    const T* coeff_array = coeffs.data() + index * cstride;
    Ae.resize(num_rows * num_cols);
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeff_array, constants.data(), coordinate_dofs.data(),
           nullptr, nullptr);
    dof_transform(Ae, cell_info, c, num_cols);
    dof_transform_to_transpose(Ae, cell_info, c, num_rows);

    // Size data structure for assembly
    be.resize(num_rows);
    std::fill(be.begin(), be.end(), 0);
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      if constexpr (_bs1 > 0)
      {
        for (int k = 0; k < _bs1; ++k)
        {
          const std::int32_t jj = _bs1 * dmap1[j] + k;
          assert(jj < (int)bc_markers1.size());
          if (bc_markers1[jj])
          {
            const T bc = bc_values1[jj];
            const T _x0 = x0.empty() ? 0.0 : x0[jj];
            // const T _x0 = 0.0;
            // be -= Ae.col(bs1 * j + k) * scale * (bc - _x0);
            for (int m = 0; m < num_rows; ++m)
              be[m] -= Ae[m * num_cols + _bs1 * j + k] * scale * (bc - _x0);
          }
        }
      }
      else
      {
        for (int k = 0; k < bs1; ++k)
        {
          const std::int32_t jj = bs1 * dmap1[j] + k;
          assert(jj < (int)bc_markers1.size());
          if (bc_markers1[jj])
          {
            const T bc = bc_values1[jj];
            const T _x0 = x0.empty() ? 0.0 : x0[jj];
            // be -= Ae.col(bs1 * j + k) * scale * (bc - _x0);
            for (int m = 0; m < num_rows; ++m)
              be[m] -= Ae[m * num_cols + bs1 * j + k] * scale * (bc - _x0);
          }
        }
      }
    }

    for (std::size_t i = 0; i < dmap0.size(); ++i)
    {
      if constexpr (_bs0 > 0)
      {
        for (int k = 0; k < _bs0; ++k)
          b[_bs0 * dmap0[i] + k] += be[_bs0 * i + k];
      }
      else
      {
        for (int k = 0; k < bs0; ++k)
          b[bs0 * dmap0[i] + k] += be[bs0 * i + k];
      }
    }
  }
}

/// @tparam T The scalar type
/// @tparam _bs0 The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs0` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @tparam _bs1 The block size of the trial function dof map.
template <typename T, int _bs = -1>
void _lift_bc_exterior_facets(
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, FEkernel<T> auto kernel,
    std::span<const std::int32_t> facets,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    mdspan2_t dofmap0, int bs0,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    mdspan2_t dofmap1, int bs1, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info, std::span<const T> bc_values1,
    std::span<const std::int8_t> bc_markers1, std::span<const T> x0, T scale)
{
  if (facets.empty())
    return;

  // Data structures used in bc application
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> Ae, be;
  assert(facets.size() % 2 == 0);
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    std::int32_t cell = facets[index];
    std::int32_t local_facet = facets[index + 1];

    // Get dof maps for cell
    auto dmap1 = stdex::submdspan(dofmap1, cell, stdex::full_extent);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        if (bc_markers1[bs1 * dmap1[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }

    if (!has_bc)
      continue;

    // Get cell coordinates/geometry
    auto x_dofs = stdex::submdspan(x_dofmap, cell, stdex::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Size data structure for assembly
    auto dmap0 = stdex::submdspan(dofmap0, cell, stdex::full_extent);

    const int num_rows = bs0 * dmap0.size();
    const int num_cols = bs1 * dmap1.size();

    const T* coeff_array = coeffs.data() + index / 2 * cstride;
    Ae.resize(num_rows * num_cols);
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeff_array, constants.data(), coordinate_dofs.data(),
           &local_facet, nullptr);
    dof_transform(Ae, cell_info, cell, num_cols);
    dof_transform_to_transpose(Ae, cell_info, cell, num_rows);

    // Size data structure for assembly
    be.resize(num_rows);
    std::fill(be.begin(), be.end(), 0);
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0.0 : x0[jj];
          // be -= Ae.col(bs1 * j + k) * scale * (bc - _x0);
          for (int m = 0; m < num_rows; ++m)
            be[m] -= Ae[m * num_cols + bs1 * j + k] * scale * (bc - _x0);
        }
      }
    }

    for (std::size_t i = 0; i < dmap0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0[i] + k] += be[bs0 * i + k];
  }
}

/// @tparam T The scalar type
/// @tparam _bs0 The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs0` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @tparam _bs1 The block size of the trial function dof map.
template <typename T, int _bs = -1>
void _lift_bc_interior_facets(
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, int num_cell_facets,
    FEkernel<T> auto kernel, std::span<const std::int32_t> facets,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    mdspan2_t dofmap0, int bs0,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    mdspan2_t dofmap1, int bs1, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info,
    const std::function<std::uint8_t(std::size_t)>& get_perm,
    std::span<const T> bc_values1, std::span<const std::int8_t> bc_markers1,
    std::span<const T> x0, T scale)
{
  if (facets.empty())
    return;

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

  const int num_dofs0 = dofmap0.extent(1);
  const int num_dofs1 = dofmap1.extent(1);
  for (std::size_t index = 0; index < facets.size(); index += 4)
  {
    std::array<std::int32_t, 2> cells = {facets[index], facets[index + 2]};
    std::array<std::int32_t, 2> local_facet
        = {facets[index + 1], facets[index + 3]};

    // Get cell geometry
    auto x_dofs0 = stdex::submdspan(x_dofmap, cells[0], stdex::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = stdex::submdspan(x_dofmap, cells[1], stdex::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    // Get dof maps for cells and pack
    auto dmap0_cell0
        = std::span(dofmap0.data_handle() + cells[0] * num_dofs0, num_dofs0);
    auto dmap0_cell1
        = std::span(dofmap1.data_handle() + cells[1] * num_dofs1, num_dofs1);

    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::copy(dmap0_cell0.begin(), dmap0_cell0.end(), dmapjoint0.begin());
    std::copy(dmap0_cell1.begin(), dmap0_cell1.end(),
              std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    auto dmap1_cell0
        = std::span(dofmap1.data_handle() + cells[0] * num_dofs1, num_dofs1);
    auto dmap1_cell1
        = std::span(dofmap1.data_handle() + cells[1] * num_dofs1, num_dofs1);

    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    std::copy(dmap1_cell0.begin(), dmap1_cell0.end(), dmapjoint1.begin());
    std::copy(dmap1_cell1.begin(), dmap1_cell1.end(),
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
    std::fill(Ae.begin(), Ae.end(), 0);
    const std::array perm{
        get_perm(cells[0] * num_cell_facets + local_facet[0]),
        get_perm(cells[1] * num_cell_facets + local_facet[1])};
    kernel(Ae.data(), coeffs.data() + index / 2 * cstride, constants.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data());

    std::span<T> _Ae(Ae);
    std::span<T> sub_Ae0 = _Ae.subspan(bs0 * dmap0_cell0.size() * num_cols,
                                       bs0 * dmap0_cell1.size() * num_cols);
    std::span<T> sub_Ae1
        = _Ae.subspan(bs1 * dmap1_cell0.size(),
                      num_rows * num_cols - bs1 * dmap1_cell0.size());

    dof_transform(_Ae, cell_info, cells[0], num_cols);
    dof_transform(sub_Ae0, cell_info, cells[1], num_cols);
    dof_transform_to_transpose(_Ae, cell_info, cells[0], num_rows);
    dof_transform_to_transpose(sub_Ae1, cell_info, cells[1], num_rows);

    be.resize(num_rows);
    std::fill(be.begin(), be.end(), 0);

    // Compute b = b - A*b for cell0
    for (std::size_t j = 0; j < dmap1_cell0.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1_cell0[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0.0 : x0[jj];
          // be -= Ae.col(bs1 * j + k) * scale * (bc - _x0);
          for (int m = 0; m < num_rows; ++m)
            be[m] -= Ae[m * num_cols + bs1 * j + k] * scale * (bc - _x0);
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
          const T _x0 = x0.empty() ? 0.0 : x0[jj];
          // be -= Ae.col(offset + bs1 * j + k) * scale * (bc - x0[jj]);
          for (int m = 0; m < num_rows; ++m)
          {
            be[m]
                -= Ae[m * num_cols + offset + bs1 * j + k] * scale * (bc - _x0);
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
/// Execute kernel over cells and accumulate result in vector
/// @tparam T The scalar type
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
template <typename T, int _bs = -1>
void assemble_cells(
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    std::span<const std::int32_t> cells, mdspan2_t dofmap, int bs,
    FEkernel<T> auto kernel, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info)
{
  assert(_bs < 0 or _bs == bs);

  if (cells.empty())
    return;

  // FIXME: Add proper interface for num_dofs
  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> be(bs * dofmap.extent(1));
  std::span<T> _be(be);

  // Iterate over active cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];

    // Get cell coordinates/geometry
    auto x_dofs = stdex::submdspan(x_dofmap, c, stdex::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate vector for cell
    std::fill(be.begin(), be.end(), 0);
    kernel(be.data(), coeffs.data() + index * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    dof_transform(_be, cell_info, c, 1);

    // Scatter cell vector to 'global' vector array
    auto dofs = stdex::submdspan(dofmap, c, stdex::full_extent);
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

/// Execute kernel over cells and accumulate result in vector
/// @tparam T The scalar type
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
template <typename T, int _bs = -1>
void assemble_exterior_facets(
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    std::span<const std::int32_t> facets, mdspan2_t dofmap, int bs,
    FEkernel<T> auto fn, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info)
{
  assert(_bs < 0 or _bs == bs);

  if (facets.empty())
    return;

  // FIXME: Add proper interface for num_dofs
  // Create data structures used in assembly
  const int num_dofs = dofmap.extent(1);
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  std::vector<T> be(bs * num_dofs);
  std::span<T> _be(be);
  assert(facets.size() % 2 == 0);
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    std::int32_t cell = facets[index];
    std::int32_t local_facet = facets[index + 1];

    // Get cell coordinates/geometry
    auto x_dofs = stdex::submdspan(x_dofmap, cell, stdex::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate element vector
    std::fill(be.begin(), be.end(), 0);
    fn(be.data(), coeffs.data() + index / 2 * cstride, constants.data(),
       coordinate_dofs.data(), &local_facet, nullptr);

    dof_transform(_be, cell_info, cell, 1);

    // Add element vector to global vector
    auto dofs = stdex::submdspan(dofmap, cell, stdex::full_extent);
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

/// Assemble linear form interior facet integrals into an vector
/// @tparam T The scalar type
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
template <typename T, int _bs = -1>
void assemble_interior_facets(
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    std::span<T> b, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, int num_cell_facets,
    std::span<const std::int32_t> facets, const fem::DofMap& dofmap,
    FEkernel<T> auto fn, std::span<const T> constants,
    std::span<const T> coeffs, int cstride,
    std::span<const std::uint32_t> cell_info,
    const std::function<std::uint8_t(std::size_t)>& get_perm)
{
  // Create data structures used in assembly
  using X = scalar_value_type_t<T>;
  std::vector<X> coordinate_dofs(2 * x_dofmap.extent(1) * 3);
  std::span<X> cdofs0(coordinate_dofs.data(), x_dofmap.extent(1) * 3);
  std::span<X> cdofs1(coordinate_dofs.data() + x_dofmap.extent(1) * 3,
                      x_dofmap.extent(1) * 3);
  std::vector<T> be;

  const int bs = dofmap.bs();
  assert(_bs < 0 or _bs == bs);
  assert(facets.size() % 4 == 0);
  for (std::size_t index = 0; index < facets.size(); index += 4)
  {
    std::array<std::int32_t, 2> cells = {facets[index], facets[index + 2]};
    std::array<std::int32_t, 2> local_facet
        = {facets[index + 1], facets[index + 3]};

    // Get cell geometry
    auto x_dofs0 = stdex::submdspan(x_dofmap, cells[0], stdex::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = stdex::submdspan(x_dofmap, cells[1], stdex::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    // Get dofmaps for cells
    std::span<const std::int32_t> dmap0 = dofmap.cell_dofs(cells[0]);
    std::span<const std::int32_t> dmap1 = dofmap.cell_dofs(cells[1]);

    // Tabulate element vector
    be.resize(bs * (dmap0.size() + dmap1.size()));
    std::fill(be.begin(), be.end(), 0);
    const std::array perm{
        get_perm(cells[0] * num_cell_facets + local_facet[0]),
        get_perm(cells[1] * num_cell_facets + local_facet[1])};
    fn(be.data(), coeffs.data() + index / 2 * cstride, constants.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data());

    std::span<T> _be(be);
    std::span<T> sub_be = _be.subspan(bs * dmap0.size(), bs * dmap1.size());

    dof_transform(be, cell_info, cells[0], 1);
    dof_transform(sub_be, cell_info, cells[1], 1);

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
/// b <- b - scale * A (x_bc - x0)
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
/// @param[in] scale Scaling to apply
template <typename T, std::floating_point U>
void lift_bc(std::span<T> b, const Form<T, U>& a, mdspan2_t x_dofmap,
             std::span<const scalar_value_type_t<T>> x,
             std::span<const T> constants,
             const std::map<std::pair<IntegralType, int>,
                            std::pair<std::span<const T>, int>>& coefficients,
             std::span<const T> bc_values1,
             std::span<const std::int8_t> bc_markers1, std::span<const T> x0,
             T scale)
{
  std::shared_ptr<const mesh::Mesh<U>> mesh = a.mesh();
  assert(mesh);

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

  const bool needs_transformation_data
      = element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a.needs_facet_permutations();

  std::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform = element0->template get_dof_transformation_function<T>();
  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform_to_transpose
      = element1->template get_dof_transformation_to_transpose_function<T>();

  for (int i : a.integral_ids(IntegralType::cell))
  {
    auto kernel = a.kernel(IntegralType::cell, i);
    assert(kernel);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = a.domain(IntegralType::cell, i);
    if (bs0 == 1 and bs1 == 1)
    {
      _lift_bc_cells<T, 1, 1>(b, x_dofmap, x, kernel, cells, dof_transform,
                              dofmap0, bs0, dof_transform_to_transpose, dofmap1,
                              bs1, constants, coeffs, cstride, cell_info,
                              bc_values1, bc_markers1, x0, scale);
    }
    else if (bs0 == 3 and bs1 == 3)
    {
      _lift_bc_cells<T, 3, 3>(b, x_dofmap, x, kernel, cells, dof_transform,
                              dofmap0, bs0, dof_transform_to_transpose, dofmap1,
                              bs1, constants, coeffs, cstride, cell_info,
                              bc_values1, bc_markers1, x0, scale);
    }
    else
    {
      _lift_bc_cells(b, x_dofmap, x, kernel, cells, dof_transform, dofmap0, bs0,
                     dof_transform_to_transpose, dofmap1, bs1, constants,
                     coeffs, cstride, cell_info, bc_values1, bc_markers1, x0,
                     scale);
    }
  }

  for (int i : a.integral_ids(IntegralType::exterior_facet))
  {
    auto kernel = a.kernel(IntegralType::exterior_facet, i);
    assert(kernel);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});
    _lift_bc_exterior_facets(
        b, x_dofmap, x, kernel, a.domain(IntegralType::exterior_facet, i),
        dof_transform, dofmap0, bs0, dof_transform_to_transpose, dofmap1, bs1,
        constants, coeffs, cstride, cell_info, bc_values1, bc_markers1, x0,
        scale);
  }

  if (a.num_integrals(IntegralType::interior_facet) > 0)
  {
    std::function<std::uint8_t(std::size_t)> get_perm;
    if (a.needs_facet_permutations())
    {
      mesh->topology_mutable()->create_entity_permutations();
      const std::vector<std::uint8_t>& perms
          = mesh->topology()->get_facet_permutations();
      get_perm = [&perms](std::size_t i) { return perms[i]; };
    }
    else
      get_perm = [](std::size_t) { return 0; };

    auto cell_types = mesh->topology()->cell_types();
    if (cell_types.size() > 1)
      throw std::runtime_error("Multiple cell types in the assembler");
    int num_cell_facets = mesh::cell_num_entities(cell_types.back(),
                                                  mesh->topology()->dim() - 1);
    for (int i : a.integral_ids(IntegralType::interior_facet))
    {
      auto kernel = a.kernel(IntegralType::interior_facet, i);
      assert(kernel);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      _lift_bc_interior_facets(
          b, x_dofmap, x, num_cell_facets, kernel,
          a.domain(IntegralType::interior_facet, i), dof_transform, dofmap0,
          bs0, dof_transform_to_transpose, dofmap1, bs1, constants, coeffs,
          cstride, cell_info, get_perm, bc_values1, bc_markers1, x0, scale);
    }
  }
}

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) row index. For a non-blocked problem j = 0.
/// The boundary conditions bc1 are on the trial spaces V_j. The forms
/// in [a] must have the same test space as L (from which b was built),
/// but the trial space may differ. If x0 is not supplied, then it is
/// treated as zero.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where a[j] is the form that
/// generates A_j
/// @param[in] x_dofmap Mesh geometry dofmap
/// @param[in] x Mesh coordinates
/// @param[in] constants Constants that appear in `a`
/// @param[in] coeffs Coefficients that appear in `a`
/// @param[in] bcs1 List of boundary conditions for each block, i.e.
/// bcs1[2] are the boundary conditions applied to the columns of a[2] /
/// x0[2] block
/// @param[in] x0 The vectors used in the lifting
/// @param[in] scale Scaling to apply
template <typename T, std::floating_point U>
void apply_lifting(
    std::span<T> b, const std::vector<std::shared_ptr<const Form<T, U>>> a,
    mdspan2_t x_dofmap, std::span<const scalar_value_type_t<T>> x,
    const std::vector<std::span<const T>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const T>, int>>>& coeffs,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC<T, U>>>>&
        bcs1,
    const std::vector<std::span<const T>>& x0, T scale)
{
  // FIXME: make changes to reactivate this check
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
      assert(a[j]->function_spaces().at(0));

      auto V1 = a[j]->function_spaces()[1];
      assert(V1);
      auto map1 = V1->dofmap()->index_map;
      const int bs1 = V1->dofmap()->index_map_bs();
      assert(map1);
      const int crange = bs1 * (map1->size_local() + map1->num_ghosts());
      bc_markers1.assign(crange, false);
      bc_values1.assign(crange, 0.0);
      for (const std::shared_ptr<const DirichletBC<T, U>>& bc : bcs1[j])
      {
        bc->mark_dofs(bc_markers1);
        bc->dof_values(bc_values1);
      }

      if (!x0.empty())
      {
        lift_bc<T>(b, *a[j], x_dofmap, x, constants[j], coeffs[j], bc_values1,
                   bc_markers1, x0[j], scale);
      }
      else
      {
        lift_bc<T>(b, *a[j], x_dofmap, x, constants[j], coeffs[j], bc_values1,
                   bc_markers1, std::span<const T>(), scale);
      }
    }
  }
}

/// Assemble linear form into a vector
/// @param[in,out] b The vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] x_dofmap Mesh geometry dofmap
/// @param[in] x Mesh coordinates
/// @param[in] constants Packed constants that appear in `L`
/// @param[in] coefficients Packed coefficients that appear in `L`
template <typename T, std::floating_point U>
void assemble_vector(
    std::span<T> b, const Form<T, U>& L, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh<U>> mesh = L.mesh();
  assert(mesh);

  // Get dofmap data
  assert(L.function_spaces().at(0));
  auto element = L.function_spaces().at(0)->element();
  assert(element);
  std::shared_ptr<const fem::DofMap> dofmap
      = L.function_spaces().at(0)->dofmap();
  assert(dofmap);
  auto dofs = dofmap->map();
  const int bs = dofmap->bs();

  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform = element->template get_dof_transformation_function<T>();

  const bool needs_transformation_data
      = element->needs_dof_transformations() or L.needs_facet_permutations();
  std::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  for (int i : L.integral_ids(IntegralType::cell))
  {
    auto fn = L.kernel(IntegralType::cell, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = L.domain(IntegralType::cell, i);
    if (bs == 1)
    {
      impl::assemble_cells<T, 1>(dof_transform, b, x_dofmap, x, cells, dofs, bs,
                                 fn, constants, coeffs, cstride, cell_info);
    }
    else if (bs == 3)
    {
      impl::assemble_cells<T, 3>(dof_transform, b, x_dofmap, x, cells, dofs, bs,
                                 fn, constants, coeffs, cstride, cell_info);
    }
    else
    {
      impl::assemble_cells(dof_transform, b, x_dofmap, x, cells, dofs, bs, fn,
                           constants, coeffs, cstride, cell_info);
    }
  }

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
      impl::assemble_exterior_facets<T, 1>(dof_transform, b, x_dofmap, x,
                                           facets, dofs, bs, fn, constants,
                                           coeffs, cstride, cell_info);
    }
    else if (bs == 3)
    {
      impl::assemble_exterior_facets<T, 3>(dof_transform, b, x_dofmap, x,
                                           facets, dofs, bs, fn, constants,
                                           coeffs, cstride, cell_info);
    }
    else
    {
      impl::assemble_exterior_facets(dof_transform, b, x_dofmap, x, facets,
                                     dofs, bs, fn, constants, coeffs, cstride,
                                     cell_info);
    }
  }

  if (L.num_integrals(IntegralType::interior_facet) > 0)
  {
    std::function<std::uint8_t(std::size_t)> get_perm;
    if (L.needs_facet_permutations())
    {
      mesh->topology_mutable()->create_entity_permutations();
      const std::vector<std::uint8_t>& perms
          = mesh->topology()->get_facet_permutations();
      get_perm = [&perms](std::size_t i) { return perms[i]; };
    }
    else
      get_perm = [](std::size_t) { return 0; };

    auto cell_types = mesh->topology()->cell_types();
    if (cell_types.size() > 1)
      throw std::runtime_error("Multiple cell types in the assembler");
    int num_cell_facets = mesh::cell_num_entities(cell_types.back(),
                                                  mesh->topology()->dim() - 1);
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
            dof_transform, b, x_dofmap, x, num_cell_facets, facets, *dofmap, fn,
            constants, coeffs, cstride, cell_info, get_perm);
      }
      else if (bs == 3)
      {
        impl::assemble_interior_facets<T, 3>(
            dof_transform, b, x_dofmap, x, num_cell_facets, facets, *dofmap, fn,
            constants, coeffs, cstride, cell_info, get_perm);
      }
      else
      {
        impl::assemble_interior_facets(
            dof_transform, b, x_dofmap, x, num_cell_facets, facets, *dofmap, fn,
            constants, coeffs, cstride, cell_info, get_perm);
      }
    }
  }
}

/// @brief Assemble linear form into a vector
/// @param[in,out] b The vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] constants Packed constants that appear in `L`
/// @param[in] coefficients Packed coefficients that appear in `L`
template <typename T, std::floating_point U>
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
