// Copyright (C) 2018-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "Form.h"
#include "FunctionSpace.h"
#include "traits.h"
#include "utils.h"
#include <algorithm>
#include <concepts>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <iterator>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace dolfinx::fem::impl
{
bool has_bc(auto& dofs, auto& bc, auto bs)
{
  for (auto dof : dofs)
    for (int k = 0; k < bs; ++k)
      if (bc[bs * dof + k])
        return true;
  return false;
};

/// @brief Typedef
using mdspan2_t = md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>;

/// @brief Execute kernel over cells and accumulate result in a matrix.
///
/// @note This function must not perform any dynamic (heap) memory
/// allocation. It may be called over only a small number of cells, so
/// a per-call allocation would not be amortized. Buffers must be
/// sized by the caller and passed in via `Ab`/`cdofs_b`.
///
/// @tparam LiftingMode Selects between matrix assembly and Dirichlet
/// lifting semantics for this kernel-execution loop (see
/// fem::impl::lift_bc). When `false` (default): standard assembly --
/// the element tensor's `bc0`-marked rows and `bc1`-marked columns are
/// zeroed in-place before being passed to `mat_set`. When `true`: cells
/// with no `bc1`-marked column dofs are skipped, since they cannot
/// contribute a lifting term, and the unmodified element tensor
/// (including BC-marked columns) is passed to `mat_set`.
/// @tparam T Matrix/form scalar type.
/// @tparam U Geometry type.
/// @param mat_set Function that accumulates computed entries into a
/// matrix.
/// @param[in] x_dofmap Degree-of-freedom map for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
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
/// in-place to the computed tensor `A` to transform trial
/// degrees-of-freedom.
/// @param bc0 Marker for rows with Dirichlet boundary conditions
/// applied.
/// @param bc1 Marker for columns with Dirichlet boundary conditions
/// applied.
/// @param kernel Kernel function to execute over each cell.
/// @param[in] coeffs Coefficient data in the kernel. It has shape
/// `(cells.size(), num_cell_coeffs)`. `coeffs(i, j)` is the `j`th
/// coefficient for cell `i`.
/// @param constants Constant data.
/// @param cell_info0 Cell permutation information for the test
/// function mesh.
/// @param cell_info1 Cell permutation information for the trial
/// function mesh.
/// @param Ab Buffer for local element matrix. Size must be at least
/// `(bs0 * num_dofs0) * (bs1 * num_dofs1)`, where `bs0 * num_dofs0` is
/// the number of rows and `bs1 * num_dofs1` is the number of columns in
/// local element matrix.
/// @param cdofs_b Buffer for local element geometry. Size must be at
/// least `3 * x_dofmap.extent(1))`.
template <bool LiftingMode, dolfinx::scalar T, std::floating_point U>
void assemble_cells_matrix(
    la::MatSet<T> auto mat_set, MDSpan2Int32 auto x_dofmap,
    MDSpan2Floating<U> auto x, std::span<const std::int32_t> cells,
    const DofMapPackCells auto& dofmap0,
    const fem::DofTransformKernel<T> auto& P0,
    const DofMapPackCells auto& dofmap1,
    const fem::DofTransformKernel<T> auto& P1T,
    std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1,
    const FEkernel<T, U> auto& kernel,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1, std::span<T> Ab,
    std::span<U> cdofs_b)
{
  if (cells.empty())
    return;

  const auto [dmap0, bs0, cells0] = dofmap0;
  const auto [dmap1, bs1, cells1] = dofmap1;

  std::size_t num_dofs0 = dmap0.extent(1);
  std::size_t num_dofs1 = dmap1.extent(1);
  std::size_t ndim0 = bs0 * num_dofs0;
  std::size_t ndim1 = bs1 * num_dofs1;

  const U* x_ptr = x.data_handle();
  const std::int32_t gdim = x.extent(1);
  const std::int32_t* x_dofmap_ptr = x_dofmap.data_handle();
  const std::int32_t num_x_dofs_cell = x_dofmap.extent(1);

  assert(Ab.size() >= ndim0 * ndim1);
  assert(cdofs_b.size() >= 3 * x_dofmap.extent(1));
  auto Ae = Ab.first(ndim0 * ndim1);

  // P0/P1T do not change across cells in this call, so whether each is a
  // set (non-null) transform is loop-invariant -- checked once here
  // rather than on every cell.
  const bool p0_set = is_transform_set(P0);
  const bool p1t_set = is_transform_set(P1T);

  const T* coeffs_data = coeffs.data_handle();
  const std::size_t cstride = coeffs.extent(1);

  // Iterate over active cells
  assert(cells0.size() == cells.size());
  assert(cells1.size() == cells.size());
  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    // Cell index in integration domain mesh (c), test function mesh
    // (c0) and trial function mesh (c1)
    std::int32_t cell = cells[c];
    std::int32_t cell0 = cells0[c];
    std::int32_t cell1 = cells1[c];

    std::span dofs0(dmap0.data_handle() + cell0 * num_dofs0, num_dofs0);
    std::span dofs1(dmap1.data_handle() + cell1 * num_dofs1, num_dofs1);

    // In "LiftingMode" only execute kernel if there are BCs on column space
    if constexpr (LiftingMode)
    {
      if (!has_bc(dofs1, bc1, bs1))
        continue;
    }

    // Get cell coordinates/geometry
    for (std::int32_t i = 0; i < num_x_dofs_cell; ++i)
    {
      const U* _x_ptr = x_ptr + x_dofmap_ptr[cell * num_x_dofs_cell + i] * gdim;
      std::copy_n(_x_ptr, gdim, cdofs_b.data() + 3 * i);
    }

    // Tabulate tensor
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), coeffs_data + c * cstride, constants.data(),
           cdofs_b.data(), nullptr, nullptr, nullptr);

    // Compute A = P_0 \tilde{A} P_1^T (dof transformation)
    if (p0_set)
      P0(Ae, cell_info0, cell0, ndim1); // B = P0 \tilde{A}
    if (p1t_set)
      P1T(Ae, cell_info1, cell1, ndim0); // A =  B P1_T

    // In lifting mode only BC dofs are assembled, while in standard mode these
    // row/column dofs are zeroed.
    if constexpr (!LiftingMode)
    {
      // Zero rows and columns for BCs
      if (!bc0.empty())
      {
        for (std::size_t i = 0; i < num_dofs0; ++i)
        {
          for (int k = 0; k < bs0; ++k)
          {
            if (bc0[bs0 * dofs0[i] + k])
            {
              // Zero row bs0 * i + k
              const int row = bs0 * i + k;
              std::fill_n(std::next(Ae.begin(), ndim1 * row), ndim1, 0);
            }
          }
        }
      }

      if (!bc1.empty())
      {
        for (std::size_t j = 0; j < num_dofs1; ++j)
        {
          for (int k = 0; k < bs1; ++k)
          {
            if (bc1[bs1 * dofs1[j] + k])
            {
              // Zero column bs1 * j + k
              int col = bs1 * j + k;
              for (std::size_t row = 0; row < ndim0; ++row)
                Ae[row * ndim1 + col] = 0;
            }
          }
        }
      }
    }

    mat_set(dofs0, dofs1, Ae);
  }
}

/// @brief Execute kernel over entities of codimension ≥ 1 and
/// accumulate result in a matrix.
///
/// Each entity is represented by (i) a cell that the entity is attached
/// to and (ii) the local index of the entity  with respect to the cell.
/// The kernel is executed for each entity. The kernel can access data
/// (e.g., coefficients, basis functions) associated with the attached
/// cell. However, entities may be attached to more than one cell. This
/// function therefore computes 'one-sided' integrals, i.e. evaluates
/// integrals as seen from cell used to define the entity.
///
/// @note This function must not perform any dynamic (heap) memory
/// allocation. It may be called over only a small number of entities,
/// so a per-call allocation would not be amortized. Buffers must be
/// sized by the caller and passed in via `Ab`/`cdofs_b`.
///
/// @tparam LiftingMode Selects between matrix assembly and Dirichlet
/// lifting semantics for this kernel-execution loop (see
/// fem::impl::lift_bc). When `false` (default): standard assembly --
/// the element tensor's `bc0`-marked rows and `bc1`-marked columns are
/// zeroed in-place before being passed to `mat_set`. When `true`: cells
/// with no `bc1`-marked column dofs are skipped, since they cannot
/// contribute a lifting term, and the unmodified element tensor
/// (including BC-marked columns) is passed to `mat_set`.
/// @tparam T Matrix/form scalar type.
/// @tparam U Geometry type.
/// @param[in] mat_set Function that accumulates computed entries into a
/// matrix.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] entities Integration entities (in the integration domain mesh) to
/// execute the kernel over. These are pairs (cell, local entity index)
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P0 Function that applies transformation P0.A in-place to
/// transform test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P1T Function that applies transformation A.P1^T in-place
/// to transform trial degrees-of-freedom.
/// @param[in] bc0 Marker for rows with Dirichlet boundary conditions
/// applied.
/// @param[in] bc1 Marker for columns with Dirichlet boundary conditions
/// applied.
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] coeffs Coefficient data array of shape `(cells.size(),
/// cstride)`.
/// @param[in] constants Constant data.
/// @param[in] cell_info0 Cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 Cell permutation information for the trial
/// function mesh.
/// @param[in] perms Entity permutation integer. Empty if entity
/// permutations are not required.
/// @param Ab Buffer for local element matrix. Size must be at least
/// `(bs0 * num_dofs0) * (bs1 * num_dofs1)`, where `bs0 * num_dofs0` is
/// the number of rows and `bs1 * num_dofs1` is the number of columns in
/// local element matrix.
/// @param cdofs_b Buffer for local element geometry. Size must be at
/// least `3 * x_dofmap.extent(1))`.
template <bool LiftingMode, dolfinx::scalar T, std::floating_point U>
void assemble_entities(
    la::MatSet<T> auto mat_set, MDSpan2Int32 auto x_dofmap,
    MDSpan2Floating<U> auto x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2>>
        entities,
    const DofMapPackEntities auto& dofmap0,
    const fem::DofTransformKernel<T> auto& P0,
    const DofMapPackEntities auto& dofmap1,
    const fem::DofTransformKernel<T> auto& P1T,
    std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1,
    const FEkernel<T, U> auto& kernel,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<T> Ab, std::span<U> cdofs_b)
{
  if (entities.empty())
    return;

  const auto [dmap0, bs0, entities0] = dofmap0;
  const auto [dmap1, bs1, entities1] = dofmap1;

  std::size_t num_dofs0 = dmap0.extent(1);
  std::size_t num_dofs1 = dmap1.extent(1);
  std::size_t ndim0 = bs0 * num_dofs0;
  std::size_t ndim1 = bs1 * num_dofs1;
  assert(entities0.size() == entities.size());
  assert(entities1.size() == entities.size());
  assert(Ab.size() >= ndim0 * ndim1);
  assert(cdofs_b.size() >= 3 * x_dofmap.extent(1));
  auto Ae = Ab.first(ndim0 * ndim1);

  const U* x_ptr = x.data_handle();
  const std::int32_t gdim = x.extent(1);
  const std::int32_t* x_dofmap_ptr = x_dofmap.data_handle();
  const std::int32_t num_x_dofs_cell = x_dofmap.extent(1);

  // P0/P1T do not change across entities in this call, so whether each is a
  // set (non-null) transform is loop-invariant -- checked once here rather
  // than on every entity.
  const bool p0_set = is_transform_set(P0);
  const bool p1t_set = is_transform_set(P1T);

  const T* coeffs_data = coeffs.data_handle();
  const std::size_t cstride = coeffs.extent(1);

  for (std::size_t f = 0; f < entities.extent(0); ++f)
  {
    // Cell in the integration domain, local entity index relative to the
    // integration domain cell, and cells in the test and trial function
    // meshes
    std::int32_t cell = entities(f, 0);
    std::int32_t local_entity = entities(f, 1);
    std::int32_t cell0 = entities0(f, 0);
    std::int32_t cell1 = entities1(f, 0);

    std::span dofs0(dmap0.data_handle() + cell0 * num_dofs0, num_dofs0);
    std::span dofs1(dmap1.data_handle() + cell1 * num_dofs1, num_dofs1);

    // Check for BCs on column space
    if constexpr (LiftingMode)
    {
      if (!has_bc(dofs1, bc1, bs1))
        continue;
    }

    // Get cell coordinates/geometry
    for (std::int32_t i = 0; i < num_x_dofs_cell; ++i)
    {
      const U* _x_ptr = x_ptr + x_dofmap_ptr[cell * num_x_dofs_cell + i] * gdim;
      std::copy_n(_x_ptr, gdim, cdofs_b.data() + 3 * i);
    }

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_entity);

    // Tabulate tensor
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), coeffs_data + f * cstride, constants.data(),
           cdofs_b.data(), &local_entity, &perm, nullptr);
    if (p0_set)
      P0(Ae, cell_info0, cell0, ndim1);
    if (p1t_set)
      P1T(Ae, cell_info1, cell1, ndim0);

    // Don't clear rows/cols in LiftingMode
    if constexpr (!LiftingMode)
    {
      // Zero rows and columns for BCs
      if (!bc0.empty())
      {
        for (std::size_t i = 0; i < num_dofs0; ++i)
        {
          for (int k = 0; k < bs0; ++k)
          {
            if (bc0[bs0 * dofs0[i] + k])
            {
              // Zero row bs0 * i + k
              const int row = bs0 * i + k;
              std::fill_n(std::next(Ae.begin(), ndim1 * row), ndim1, 0);
            }
          }
        }
      }

      if (!bc1.empty())
      {
        for (std::size_t j = 0; j < num_dofs1; ++j)
        {
          for (int k = 0; k < bs1; ++k)
          {
            if (bc1[bs1 * dofs1[j] + k])
            {
              // Zero column bs1 * j + k
              int col = bs1 * j + k;
              for (std::size_t row = 0; row < ndim0; ++row)
                Ae[row * ndim1 + col] = 0;
            }
          }
        }
      }
    }

    mat_set(dofs0, dofs1, Ae);
  }
}

/// @brief Execute kernel over interior facets and accumulate result in
/// a matrix.
///
/// @note This function must not perform any dynamic (heap) memory
/// allocation. It may be called over only a small number of facets,
/// so a per-call allocation would not be amortized. Buffers must be
/// sized by the caller and passed in via `Ab`/`cdofs_b`/`dofs_b`.
///
/// @tparam T Matrix/form scalar type.
/// @tparam U Geometry type.
/// @tparam LiftingMode Selects between matrix assembly and Dirichlet
/// lifting semantics for this kernel-execution loop (see
/// fem::impl::lift_bc). When `false` (default): standard assembly --
/// the element tensor's `bc0`-marked rows and `bc1`-marked columns are
/// zeroed in-place before being passed to `mat_set`. When `true`: cells
/// with no `bc1`-marked column dofs are skipped, since they cannot
/// contribute a lifting term, and the unmodified element tensor
/// (including BC-marked columns) is passed to `mat_set`.
/// @param mat_set Function that accumulates computed entries into a
/// matrix.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] facets Facet indices (in the integration domain mesh) to
/// execute the kernel over.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices. Cells that don't exist in the test function domain should be
/// marked with -1 in the cell indices list.
/// @param[in] P0 Function that applies transformation P0.A in-place to
/// transform test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices. Cells that don't exist in the trial function domain should be
/// marked with -1 in the cell indices list.
/// @param[in] P1T Function that applies transformation A.P1^T in-place
/// to transform trial degrees-of-freedom.
/// @param[in] bc0 Marker for rows with Dirichlet boundary conditions
/// applied.
/// @param[in] bc1 Marker for columns with Dirichlet boundary conditions
/// applied.
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] coeffs  The coefficient data array of shape (cells.size(),
/// cstride).
/// @param[in] constants Constant data.
/// @param[in] cell_info0 Cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 Cell permutation information for the trial
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
/// @param Ab Buffer for local element matrix. Size must be at least `4
/// * (bs0 * num_dofs0) * (bs1 * num_dofs1)`, where `bs0 * num_dofs0` is
/// the number of rows and `bs1 * num_dofs1` is the number of columns in
/// local element matrix.
/// @param cdofs_b Buffer for local element geometry. Size must be at
/// least `2 * 3 * x_dofmap.extent(1))`.
/// @param dofs_b Buffer for degrees-of-freedom. Size must be at least
/// `2 * dmap0.extent(1) + 2 * dmap1.extent(1)`.
/// @param Ae_block_b Buffer used to gather a single (test, trial) block
/// of the local element matrix. Size must be at least `(bs0 *
/// dmap0.extent(1)) * (bs1 * dmap1.extent(1))`.
template <bool LiftingMode, dolfinx::scalar T, std::floating_point U>
void assemble_interior_facets(
    la::MatSet<T> auto mat_set, MDSpan2Int32 auto x_dofmap,
    MDSpan2Floating<U> auto x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2, 2>>
        facets,
    const DofMapPackFacets auto& dofmap0,
    const fem::DofTransformKernel<T> auto& P0,
    const DofMapPackFacets auto& dofmap1,
    const fem::DofTransformKernel<T> auto& P1T,
    std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1,
    const FEkernel<T, U> auto& kernel,
    md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                    md::dynamic_extent>>
        coeffs,
    std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<T> Ab, std::span<U> cdofs_b, std::span<std::int32_t> dofs_b,
    std::span<T> Ae_block_b)
{
  if (facets.empty())
    return;

  const auto [dmap0, bs0, facets0] = dofmap0;
  const auto [dmap1, bs1, facets1] = dofmap1;

  // Data structures used in assembly
  assert(cdofs_b.size() >= 2 * 3 * x_dofmap.extent(1));
  auto cdofs0 = cdofs_b.first(3 * x_dofmap.extent(1));
  auto cdofs1 = cdofs_b.last(3 * x_dofmap.extent(1));

  const U* x_ptr = x.data_handle();
  const std::int32_t gdim = x.extent(1);
  const std::int32_t* x_dofmap_ptr = x_dofmap.data_handle();
  const std::int32_t num_x_dofs_cell = x_dofmap.extent(1);

  std::size_t dmap0_size = dmap0.extent(1);
  std::size_t dmap1_size = dmap1.extent(1);
  std::size_t num_rows = bs0 * 2 * dmap0_size;
  std::size_t num_cols = bs1 * 2 * dmap1_size;

  // Dofmap data structures
  assert(dofs_b.size() >= (2 * dmap0_size) + (2 * dmap1_size));
  auto dmapjoint0 = dofs_b.first(2 * dmap0_size);
  auto dmapjoint1 = dofs_b.last(2 * dmap1_size);

  assert(facets0.size() == facets.size());
  assert(facets1.size() == facets.size());
  assert(Ab.size() >= num_rows * num_cols);
  auto Ae = Ab.first(num_rows * num_cols);

  // Buffer used to gather a contiguous (test, trial) block of Ae when
  // one of the two cells attached to the facet does not exist in the
  // test/trial function domain (e.g. an interface between two
  // domains) -- the sparsity pattern only holds entries for blocks
  // where both cells exist, so such blocks must be inserted
  // individually rather than as part of the full joint block.
  assert(Ae_block_b.size() >= dmap0_size * bs0 * dmap1_size * bs1);

  const T* coeffs_data = coeffs.data_handle();
  const std::size_t cstride = 2 * coeffs.extent(2);

  auto insert_block = [&Ae_block_b, &Ae, &bs0, &bs1, &num_cols,
                       &mat_set](std::span<const std::int32_t> rdofs,
                                 std::span<const std::int32_t> cdofs,
                                 std::size_t row_offset, std::size_t col_offset)
  {
    if (rdofs.empty() or cdofs.empty())
      return;
    auto Ae_block = Ae_block_b.first(rdofs.size() * bs0 * cdofs.size() * bs1);
    for (std::size_t i = 0; i < rdofs.size() * bs0; ++i)
    {
      auto row
          = std::next(Ae.begin(), (row_offset + i) * num_cols + col_offset);
      std::copy_n(row, cdofs.size() * bs1,
                  std::next(Ae_block.begin(), i * cdofs.size() * bs1));
    }
    mat_set(rdofs, cdofs, Ae_block);
  };

  // P0/P1T do not change across facets in this call, so whether each is a
  // set (non-null) transform is loop-invariant -- checked once here rather
  // than on every facet.
  const bool p0_set = is_transform_set(P0);
  const bool p1t_set = is_transform_set(P1T);

  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    // Cells in integration domain,  test function domain and trial
    // function domain
    std::array cells{facets(f, 0, 0), facets(f, 1, 0)};
    std::array cells0{facets0(f, 0, 0), facets0(f, 1, 0)};
    std::array cells1{facets1(f, 0, 0), facets1(f, 1, 0)};

    // Local facets indices
    std::array local_facet{facets(f, 0, 1), facets(f, 1, 1)};

    // Get cell geometry
    for (std::int32_t i = 0; i < num_x_dofs_cell; ++i)
    {
      const U* _x_ptr0
          = x_ptr + x_dofmap_ptr[cells[0] * num_x_dofs_cell + i] * gdim;
      std::copy_n(_x_ptr0, gdim, cdofs0.data() + 3 * i);
      const U* _x_ptr1
          = x_ptr + x_dofmap_ptr[cells[1] * num_x_dofs_cell + i] * gdim;
      std::copy_n(_x_ptr1, gdim, cdofs1.data() + 3 * i);
    }

    // Get dof maps for cells and pack
    // When integrating over interfaces between two domains, the test function
    // might only be defined on one side, so we check which cells exist in the
    // test function domain
    std::span<const std::int32_t> dmap0_cell0
        = cells0[0] >= 0
              ? std::span(dmap0.data_handle() + cells0[0] * dmap0_size,
                          dmap0_size)
              : std::span<const std::int32_t>();
    std::span<const std::int32_t> dmap0_cell1
        = cells0[1] >= 0
              ? std::span(dmap0.data_handle() + cells0[1] * dmap0_size,
                          dmap0_size)
              : std::span<const std::int32_t>();

    std::ranges::copy(dmap0_cell0, dmapjoint0.begin());
    std::ranges::copy(dmap0_cell1, std::next(dmapjoint0.begin(), dmap0_size));

    // Check which cells exist in the trial function domain
    std::span<const std::int32_t> dmap1_cell0
        = cells1[0] >= 0
              ? std::span(dmap1.data_handle() + cells1[0] * dmap1_size,
                          dmap1_size)
              : std::span<const std::int32_t>();
    std::span<const std::int32_t> dmap1_cell1
        = cells1[1] >= 0
              ? std::span(dmap1.data_handle() + cells1[1] * dmap1_size,
                          dmap1_size)
              : std::span<const std::int32_t>();

    std::ranges::copy(dmap1_cell0, dmapjoint1.begin());
    std::ranges::copy(dmap1_cell1, std::next(dmapjoint1.begin(), dmap1_size));

    // Check for BCs on column space
    if constexpr (LiftingMode)
    {
      if (!has_bc(dmapjoint1, bc1, bs1))
        continue;
    }

    // Tabulate tensor
    std::ranges::fill(Ae, 0);
    std::array perm = perms.empty()
                          ? std::array<std::uint8_t, 2>{0, 0}
                          : std::array{perms(cells[0], local_facet[0]),
                                       perms(cells[1], local_facet[1])};
    kernel(Ae.data(), coeffs_data + f * cstride, constants.data(),
           cdofs_b.data(), local_facet.data(), perm.data(), nullptr);

    // Local element layout is a 2x2 block matrix with structure
    //
    //   cell0cell0  |  cell0cell1
    //   cell1cell0  |  cell1cell1
    //
    // where each block is element tensor of size (dmap0, dmap1).

    // Only apply transformation when cells exist
    if (p0_set and cells0[0] >= 0)
      P0(Ae, cell_info0, cells0[0], num_cols);
    if (p0_set and cells0[1] >= 0)
    {
      std::span sub_Ae0(Ae.data() + bs0 * dmap0_size * num_cols,
                        bs0 * dmap0_size * num_cols);
      P0(sub_Ae0, cell_info0, cells0[1], num_cols);
    }
    if (p1t_set and cells1[0] >= 0)
      P1T(Ae, cell_info1, cells1[0], num_rows);

    if (p1t_set and cells1[1] >= 0)
    {
      for (std::size_t row = 0; row < num_rows; ++row)
      {
        // DOFs for dmap1 and cell1 are not stored contiguously in the
        // block matrix, so each row needs a separate span access
        std::span sub_Ae1(Ae.data() + row * num_cols + bs1 * dmap1_size,
                          bs1 * dmap1_size);
        P1T(sub_Ae1, cell_info1, cells1[1], 1);
      }
    }

    // Clear rows/cols if not in LiftingMode
    if constexpr (!LiftingMode)
    {
      // Zero rows and columns for BCs
      if (!bc0.empty())
      {
        for (std::size_t i = 0; i < dmapjoint0.size(); ++i)
        {
          for (int k = 0; k < bs0; ++k)
          {
            if (bc0[bs0 * dmapjoint0[i] + k])
            {
              // Zero row bs0 * i + k
              std::fill_n(std::next(Ae.begin(), num_cols * (bs0 * i + k)),
                          num_cols, 0);
            }
          }
        }
      }

      if (!bc1.empty())
      {
        for (std::size_t j = 0; j < dmapjoint1.size(); ++j)
        {
          for (int k = 0; k < bs1; ++k)
          {
            if (bc1[bs1 * dmapjoint1[j] + k])
            {
              // Zero column bs1 * j + k
              for (std::size_t m = 0; m < num_rows; ++m)
                Ae[m * num_cols + bs1 * j + k] = 0;
            }
          }
        }
      }
    }

    // The common case is that a cell exists on both sides of the
    // facet for both the test and trial function domains, in which
    // case the full joint block can be inserted in one go. Otherwise
    // (e.g. an interface between two domains), only the blocks
    // corresponding to existing (test, trial) cell pairs are present
    // in the sparsity pattern, so each must be inserted individually.
    if (cells0[0] >= 0 and cells0[1] >= 0 and cells1[0] >= 0 and cells1[1] >= 0)
      mat_set(dmapjoint0, dmapjoint1, Ae);
    else
    {
      insert_block(dmap0_cell0, dmap1_cell0, 0, 0);
      insert_block(dmap0_cell0, dmap1_cell1, 0, bs1 * dmap1_size);
      insert_block(dmap0_cell1, dmap1_cell0, bs0 * dmap0_size, 0);
      insert_block(dmap0_cell1, dmap1_cell1, bs0 * dmap0_size,
                   bs1 * dmap1_size);
    }
  }
}

/// @brief Assemble (accumulate) into a matrix.
///
/// Rows (bc0) and columns (bc1) with Dirichlet conditions are zeroed.
/// Markers (bc0 and bc1) can be empty if no Dirichlet conditions are
/// applied.
///
/// @tparam LiftingMode Selects between matrix assembly and Dirichlet
/// lifting semantics (see fem::impl::lift_bc, which instantiates this
/// function with `LiftingMode=true` to compute lifting contributions to
/// a right-hand side vector rather than to assemble a matrix). When
/// `false` (default): standard assembly -- the element tensor's
/// `bc0`-marked rows and `bc1`-marked columns are zeroed in-place
/// before being passed to `mat_set`. When `true`: cells with no
/// `bc1`-marked column dofs are skipped, since they cannot contribute a
/// lifting term, and the unmodified element tensor (including
/// BC-marked columns) is passed to `mat_set`.
/// @tparam T Scalar type.
/// @tparam U Geometry type.
/// @param[in] mat_set Function that accumulates computed entries into a
/// matrix.
/// @param[in] a Bilinear form to assemble.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] constants Constants that appear in `a`.
/// @param[in] coefficients Coefficients that appear in `a`.
/// @param bc0 Marker for rows with Dirichlet boundary conditions
/// applied.
/// @param bc1 Marker for columns with Dirichlet boundary conditions
/// applied.
template <bool LiftingMode, dolfinx::scalar T, std::floating_point U>
void assemble_matrix(
    la::MatSet<T> auto mat_set, const Form<T, U>& a,
    md::mdspan<const U, md::extents<std::size_t, md::dynamic_extent, 3>> x,
    std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients,
    std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1)
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

  // TODO: Mixed topology with exterior and interior facet integrals.
  //
  // NOTE: Can't just loop over cell types for interior facet integrals
  // because we have a kernel per combination of comparable cell types,
  // rather than one per cell type. Also, we need the dofmaps for two
  // different cell types at the same time.
  const int num_cell_types = mesh->topology()->cell_types().size();
  for (int cell_type_idx = 0; cell_type_idx < num_cell_types; ++cell_type_idx)
  {
    // Geometry dofmap and data
    mdspan2_t x_dofmap = mesh->geometry().dofmaps().at(cell_type_idx);

    // Get dofmap data
    std::shared_ptr<const fem::DofMap> dofmap0
        = a.function_spaces().at(0)->dofmaps().at(cell_type_idx);
    std::shared_ptr<const fem::DofMap> dofmap1
        = a.function_spaces().at(1)->dofmaps().at(cell_type_idx);
    assert(dofmap0);
    assert(dofmap1);
    md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> dofs0
        = dofmap0->map();
    const int bs0 = dofmap0->bs();
    md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> dofs1
        = dofmap1->map();
    const int bs1 = dofmap1->bs();

    // Buffers reused across all integral kernels for this cell type,
    // sized for the worst case (interior facets, which touch two cells).
    std::vector<T> Ab((2 * bs0 * dofs0.extent(1))
                      * (2 * bs1 * dofs1.extent(1)));
    std::vector<U> cdofs_b(2 * 3 * x_dofmap.extent(1));
    std::size_t dmap0_size = dofmap0->map().extent(1);
    std::size_t dmap1_size = dofmap1->map().extent(1);
    std::vector<std::int32_t> dmap_b((2 * dmap0_size) + (2 * dmap1_size));
    std::vector<T> Ae_block_b(dmap0_size * bs0 * dmap1_size * bs1);

    auto element0 = a.function_spaces().at(0)->elements(cell_type_idx);
    assert(element0);
    auto element1 = a.function_spaces().at(1)->elements(cell_type_idx);
    assert(element1);
    const fem::DofTransformKernel<T> auto& P0
        = element0->template dof_transformation_fn<T>(doftransform::standard);
    const fem::DofTransformKernel<T> auto& P1T
        = element1->template dof_transformation_right_fn<T>(
            doftransform::transpose);

    std::span<const std::uint32_t> cell_info0;
    std::span<const std::uint32_t> cell_info1;
    if (element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a.needs_facet_permutations())
    {
      mesh0->topology_mutable()->create_entity_permutations();
      mesh1->topology_mutable()->create_entity_permutations();
      cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
      cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
    }

    for (int i = 0; i < a.num_integrals(IntegralType::cell, cell_type_idx); ++i)
    {
      auto fn = a.kernel(IntegralType::cell, i, cell_type_idx);
      assert(fn);
      std::span cells = a.domain(IntegralType::cell, i, cell_type_idx);
      std::span cells0 = a.domain_arg(IntegralType::cell, 0, i, cell_type_idx);
      std::span cells1 = a.domain_arg(IntegralType::cell, 1, i, cell_type_idx);
      auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
      assert(cells.size() * cstride == coeffs.size());
      if (bs0 == 1 and bs1 == 1)
      {
        impl::assemble_cells_matrix<LiftingMode>(
            mat_set, x_dofmap, x, cells,
            std::tuple{dofs0, std::integral_constant<int, 1>{}, cells0}, P0,
            std::tuple{dofs1, std::integral_constant<int, 1>{}, cells1}, P1T,
            bc0, bc1, fn, md::mdspan(coeffs.data(), cells.size(), cstride),
            constants, cell_info0, cell_info1, std::span(Ab),
            std::span(cdofs_b));
      }
      else if (bs0 == 3 and bs1 == 3)
      {
        impl::assemble_cells_matrix<LiftingMode>(
            mat_set, x_dofmap, x, cells,
            std::tuple{dofs0, std::integral_constant<int, 3>{}, cells0}, P0,
            std::tuple{dofs1, std::integral_constant<int, 3>{}, cells1}, P1T,
            bc0, bc1, fn, md::mdspan(coeffs.data(), cells.size(), cstride),
            constants, cell_info0, cell_info1, std::span(Ab),
            std::span(cdofs_b));
      }
      else
      {
        impl::assemble_cells_matrix<LiftingMode>(
            mat_set, x_dofmap, x, cells, std::tuple{dofs0, bs0, cells0}, P0,
            std::tuple{dofs1, bs1, cells1}, P1T, bc0, bc1, fn,
            md::mdspan(coeffs.data(), cells.size(), cstride), constants,
            cell_info0, cell_info1, std::span(Ab), std::span(cdofs_b));
      }
    }

    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> facet_perms;
    if (a.needs_facet_permutations())
    {
      mesh::CellType cell_type = mesh->topology()->cell_types()[cell_type_idx];
      int num_facets_per_cell
          = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
      mesh->topology_mutable()->create_entity_permutations();
      const std::vector<std::uint8_t>& p
          = mesh->topology()->get_facet_permutations();
      facet_perms = md::mdspan(p.data(), p.size() / num_facets_per_cell,
                               num_facets_per_cell);
    }

    for (int i = 0;
         i < a.num_integrals(IntegralType::interior_facet, cell_type_idx); ++i)
    {
      if (num_cell_types > 1)
      {
        throw std::invalid_argument("Interior facet integrals with mixed "
                                    "topology aren't supported yet");
      }

      using mdspanx22_t
          = md::mdspan<const std::int32_t,
                       md::extents<std::size_t, md::dynamic_extent, 2, 2>>;
      using mdspanx2x_t
          = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                            md::dynamic_extent>>;

      auto fn = a.kernel(IntegralType::interior_facet, i, 0);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});

      std::span facets = a.domain(IntegralType::interior_facet, i, 0);
      std::span facets0 = a.domain_arg(IntegralType::interior_facet, 0, i, 0);
      std::span facets1 = a.domain_arg(IntegralType::interior_facet, 1, i, 0);
      assert((facets.size() / 4) * 2 * cstride == coeffs.size());
      if (bs0 == 1 and bs1 == 1)
      {
        impl::assemble_interior_facets<LiftingMode>(
            mat_set, x_dofmap, x,
            mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
            std::tuple{dofs0, std::integral_constant<int, 1>{},
                       mdspanx22_t(facets0.data(), facets0.size() / 4, 2, 2)},
            P0,
            std::tuple{dofs1, std::integral_constant<int, 1>{},
                       mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
            P1T, bc0, bc1, fn,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            constants, cell_info0, cell_info1, facet_perms, std::span(Ab),
            std::span(cdofs_b), dmap_b, std::span(Ae_block_b));
      }
      else if (bs0 == 3 and bs1 == 3)
      {
        impl::assemble_interior_facets<LiftingMode>(
            mat_set, x_dofmap, x,
            mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
            std::tuple{dofs0, std::integral_constant<int, 3>{},
                       mdspanx22_t(facets0.data(), facets0.size() / 4, 2, 2)},
            P0,
            std::tuple{dofs1, std::integral_constant<int, 3>{},
                       mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
            P1T, bc0, bc1, fn,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            constants, cell_info0, cell_info1, facet_perms, std::span(Ab),
            std::span(cdofs_b), dmap_b, std::span(Ae_block_b));
      }
      else
      {
        impl::assemble_interior_facets<LiftingMode>(
            mat_set, x_dofmap, x,
            mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
            std::tuple{dofs0, bs0,
                       mdspanx22_t(facets0.data(), facets0.size() / 4, 2, 2)},
            P0,
            std::tuple{dofs1, bs1,
                       mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
            P1T, bc0, bc1, fn,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            constants, cell_info0, cell_info1, facet_perms, std::span(Ab),
            std::span(cdofs_b), dmap_b, std::span(Ae_block_b));
      }
    }

    for (auto itg_type : {fem::IntegralType::exterior_facet,
                          fem::IntegralType::vertex, fem::IntegralType::ridge})
    {
      md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms
          = itg_type == fem::IntegralType::exterior_facet
                ? facet_perms
                : md::mdspan<const std::uint8_t,
                             md::dextents<std::size_t, 2>>{};

      for (int i = 0; i < a.num_integrals(itg_type, cell_type_idx); ++i)
      {
        if (num_cell_types > 1)
        {
          throw std::invalid_argument("Exterior facet integrals with mixed "
                                      "topology aren't supported yet");
        }

        using mdspanx2_t
            = md::mdspan<const std::int32_t,
                         md::extents<std::size_t, md::dynamic_extent, 2>>;

        auto fn = a.kernel(itg_type, i, 0);
        assert(fn);
        auto& [coeffs, cstride] = coefficients.at({itg_type, i});

        std::span e = a.domain(itg_type, i, 0);
        mdspanx2_t entities(e.data(), e.size() / 2, 2);
        std::span e0 = a.domain_arg(itg_type, 0, i, 0);
        mdspanx2_t entities0(e0.data(), e0.size() / 2, 2);
        std::span e1 = a.domain_arg(itg_type, 1, i, 0);
        mdspanx2_t entities1(e1.data(), e1.size() / 2, 2);
        assert((entities.size() / 2) * cstride == coeffs.size());
        if (bs0 == 1 and bs1 == 1)
        {
          impl::assemble_entities<LiftingMode>(
              mat_set, x_dofmap, x, entities,
              std::tuple{dofs0, std::integral_constant<int, 1>{}, entities0},
              P0,
              std::tuple{dofs1, std::integral_constant<int, 1>{}, entities1},
              P1T, bc0, bc1, fn,
              md::mdspan(coeffs.data(), entities.extent(0), cstride), constants,
              cell_info0, cell_info1, perms, std::span(Ab), std::span(cdofs_b));
        }
        else if (bs0 == 3 and bs1 == 3)
        {
          impl::assemble_entities<LiftingMode>(
              mat_set, x_dofmap, x, entities,
              std::tuple{dofs0, std::integral_constant<int, 3>{}, entities0},
              P0,
              std::tuple{dofs1, std::integral_constant<int, 3>{}, entities1},
              P1T, bc0, bc1, fn,
              md::mdspan(coeffs.data(), entities.extent(0), cstride), constants,
              cell_info0, cell_info1, perms, std::span(Ab), std::span(cdofs_b));
        }
        else
        {
          impl::assemble_entities<LiftingMode>(
              mat_set, x_dofmap, x, entities, std::tuple{dofs0, bs0, entities0},
              P0, std::tuple{dofs1, bs1, entities1}, P1T, bc0, bc1, fn,
              md::mdspan(coeffs.data(), entities.extent(0), cstride), constants,
              cell_info0, cell_info1, perms, std::span(Ab), std::span(cdofs_b));
        }
      }
    }
  }
}
} // namespace dolfinx::fem::impl
