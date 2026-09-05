// Copyright (C) 2018-2026 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "DirichletBC.h"
#include "DofMap.h"
#include "Form.h"
#include "assemble_matrix_impl.h"
#include "traits.h"
#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dolfinx::fem
{
template <dolfinx::scalar T, std::floating_point U>
class DirichletBC;
}

namespace dolfinx::fem::impl
{
/// @cond
using mdspan2_t = md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>;
/// @endcond

/// @brief Execute kernel over cells and accumulate result in vector.
///
/// @note This function must not perform any dynamic (heap) memory
/// allocation. It may be called over only a small number of cells, so
/// a per-call allocation would not be amortized. Buffers must be
/// sized by the caller and passed in via `be_b`/`cdofs_b`.
///
/// @tparam V Vector container type (i.e. the type of `b`).
/// @tparam T Scalar type.
/// @param[in] P0 Function that applies transformation `P0.b` in-place
/// to `b` to transform test degrees-of-freedom.
/// @param[in,out] b Array to accumulate into.
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
/// @param[in] be_b Buffer for local element vector. Size must be at
/// least `bs * dmap.extent(1)`.
/// @param[in] cdofs_b Buffer for local element geometry. Size must be
/// at least `3 * x_dofmap.extent(1)`.
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_cells(const fem::DofTransformKernel<T> auto& P0, V&& b,
                    MDSpan2Int32 auto x_dofmap, MDSpan2Floating<U> auto x,
                    std::span<const std::int32_t> cells,
                    const DofMapPackCells auto& dofmap,
                    const FEkernel<T, U> auto& kernel,
                    std::span<const T> constants,
                    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
                    std::span<const std::uint32_t> cell_info0,
                    std::span<T> be_b, std::span<U> cdofs_b)
{
  if (cells.empty())
    return;

  const auto& [dmap, bs, cells0] = dofmap;
  assert(cdofs_b.size() >= 3 * x_dofmap.extent(1));
  assert(be_b.size() >= bs * dmap.extent(1));
  auto be = be_b.first(bs * dmap.extent(1));

  const U* x_ptr = x.data_handle();
  const std::int32_t* x_dofmap_ptr = x_dofmap.data_handle();
  const std::int32_t* dmap_ptr = dmap.data_handle();

  // P0 does not change across cells in this call, so whether it is a
  // set (non-null) transform is loop-invariant -- checked once here
  // rather than on every cell.
  const bool p0_set = is_transform_set(P0);

  const T* coeffs_data = coeffs.data_handle();
  const std::size_t cstride = coeffs.extent(1);

  // Iterate over active cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Integration domain cell and test function cell
    std::int32_t c = cells[index];
    std::int32_t c0 = cells0[index];

    // Get cell coordinates/geometry
    for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
    {
      const U* _x_ptr
          = x_ptr + x_dofmap_ptr[c * x_dofmap.extent(1) + i] * x.extent(1);
      U* cdofs = cdofs_b.data() + 3 * i;
      for (std::size_t j = 0; j < x.extent(1); ++j)
        cdofs[j] = _x_ptr[j];
    }

    // Tabulate vector for cell
    std::ranges::fill(be, 0);
    kernel(be.data(), coeffs_data + index * cstride, constants.data(),
           cdofs_b.data(), nullptr, nullptr, nullptr);
    if (p0_set)
      P0(be, cell_info0, c0, 1);

    // Scatter cell vector to 'global' vector array
    std::span dofs(dmap_ptr + c0 * dmap.extent(1), dmap.extent(1));
    for (std::size_t i = 0; i < dmap.extent(1); ++i)
    {
      std::int32_t dof = bs * dofs[i];
      std::int32_t offset = bs * i;
      for (int k = 0; k < bs; ++k)
        b[dof + k] += be[offset + k];
    }
  }
}

/// @brief Execute kernel over entities of codimension ≥ 1 and accumulate result
/// in a matrix.
///
/// Each entity is represented by (i) a cell that the entity is attached to
/// and (ii) the local index of the entity  with respect to the cell. The
/// kernel is executed for each entity. The kernel can access data
/// (e.g., coefficients, basis functions) associated with the attached cell.
/// However, entities may be attached to more than one cell. This function
/// therefore computes 'one-sided' integrals, i.e. evaluates integrals as seen
/// from cell used to define the entity.
///
/// @note This function must not perform any dynamic (heap) memory
/// allocation. It may be called over only a small number of
/// entities, so a per-call allocation would not be amortized. Buffers
/// must be sized by the caller and passed in via `be_b`/`cdofs_b`.
///
/// @tparam V Vector container type (i.e. the type of `b`).
/// @tparam T Scalar type.
/// @param P0 Function that applies transformation `P0.b` in-place to
/// transform test degrees-of-freedom.
/// @param[in,out] b The vector to accumulate into.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] entities Entities (in the integration domain mesh) to execute
/// the kernel over.
/// @param[in] dofmap Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] constants The constant data.
/// @param[in] coeffs The coefficient data array of shape
/// `(cells.size(), coeffs_per_cell)`.
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] perms Entity permutation integer. Empty if entity
/// permutations are not required.
/// @param[in] be_b Buffer for local element vector. Size must be at
/// least `bs * dmap.extent(1)`.
/// @param[in] cdofs_b Buffer for local element geometry. Size must be
/// at least `3 * x_dofmap.extent(1)`.
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_entities(
    const fem::DofTransformKernel<T> auto& P0, V&& b,
    MDSpan2Int32 auto x_dofmap, MDSpan2Floating<U> auto x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2>>
        entities,
    const DofMapPackEntities auto& dofmap, const FEkernel<T, U> auto& kernel,
    std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const std::uint32_t> cell_info0,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<T> be_b, std::span<U> cdofs_b)
{
  if (entities.empty())
    return;

  const auto [dmap, bs, entities0] = dofmap;

  const std::size_t num_dofs = dmap.extent(1);
  assert(cdofs_b.size() >= 3 * x_dofmap.extent(1));
  assert(be_b.size() >= static_cast<std::size_t>(bs) * num_dofs);
  auto be = be_b.first(bs * num_dofs);
  assert(entities0.size() == entities.size());

  const U* x_ptr = x.data_handle();
  const std::int32_t gdim = x.extent(1);
  const std::int32_t* x_dofmap_ptr = x_dofmap.data_handle();
  const std::int32_t num_x_dofs_cell = x_dofmap.extent(1);
  const std::int32_t* dmap_ptr = dmap.data_handle();

  // P0 does not change across entities in this call, so whether it is a
  // set (non-null) transform is loop-invariant -- checked once here rather
  // than on every entity.
  const bool p0_set = is_transform_set(P0);

  const T* coeffs_data = coeffs.data_handle();
  const std::size_t cstride = coeffs.extent(1);

  for (std::size_t f = 0; f < entities.extent(0); ++f)
  {
    // Cell in the integration domain, local facet index relative to the
    // integration domain cell, and cell in the test function mesh
    std::int32_t cell = entities(f, 0);
    std::int32_t local_entity = entities(f, 1);
    std::int32_t cell0 = entities0(f, 0);

    // Get cell coordinates/geometry
    for (std::int32_t i = 0; i < num_x_dofs_cell; ++i)
    {
      const U* _x_ptr = x_ptr + x_dofmap_ptr[cell * num_x_dofs_cell + i] * gdim;
      std::copy_n(_x_ptr, gdim, cdofs_b.data() + 3 * i);
    }

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_entity);

    // Tabulate element vector
    std::ranges::fill(be, 0);
    kernel(be.data(), coeffs_data + f * cstride, constants.data(),
           cdofs_b.data(), &local_entity, &perm, nullptr);
    if (p0_set)
      P0(be, cell_info0, cell0, 1);

    // Add to global vector
    std::span dofs(dmap_ptr + cell0 * num_dofs, num_dofs);
    for (std::size_t i = 0; i < dofs.size(); ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs[i] + k] += be[bs * i + k];
  }
}

/// @brief Assemble linear form interior facet integrals into an vector.
///
/// @note This function must not perform any dynamic (heap) memory
/// allocation. It may be called over only a small number of facets,
/// so a per-call allocation would not be amortized. Buffers must be
/// sized by the caller and passed in via `be_b`/`cdofs_b`.
///
/// @tparam V Vector container type (i.e. the type of `b`).
/// @tparam T Scalar type.
/// @param P0 Function that applies transformation P0.A in-place to
/// transform trial degrees-of-freedom.
/// @param[in,out] b The vector to accumulate into.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] facets Facets (in the integration domain mesh) to execute
/// the kernel over.
/// @param[in] dofmap Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// Cells that don't exist in the test function domain should be marked
/// with -1 in the cell indices list.
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] constants The constant data
/// @param[in] coeffs Coefficient data array, withshape (cells.size(),
/// cstride).
/// @param[in] cell_info0 The cell permutation information for the test
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
/// @param[in] be_b Buffer for local element vector. Size must be at
/// least `2 * bs * dmap.extent(1)`.
/// @param[in] cdofs_b Buffer for local element geometry. Size must be
/// at least `2 * 3 * x_dofmap.extent(1)`.
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_interior_facets(
    const fem::DofTransformKernel<T> auto& P0, V&& b,
    MDSpan2Int32 auto x_dofmap, MDSpan2Floating<U> auto x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2, 2>>
        facets,
    const DofMapPackFacets auto& dofmap, const FEkernel<T, U> auto& kernel,
    std::span<const T> constants,
    md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                    md::dynamic_extent>>
        coeffs,
    std::span<const std::uint32_t> cell_info0,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<T> be_b, std::span<U> cdofs_b)
{
  if (facets.empty())
    return;

  const auto [dmap, bs, facets0] = dofmap;

  assert(cdofs_b.size() >= 2 * x_dofmap.extent(1) * 3);
  auto cdofs0 = cdofs_b.first(x_dofmap.extent(1) * 3);
  auto cdofs1 = cdofs_b.subspan(x_dofmap.extent(1) * 3, x_dofmap.extent(1) * 3);

  const std::size_t dmap_size = dmap.extent(1);
  assert(be_b.size() >= static_cast<std::size_t>(bs) * 2 * dmap_size);
  auto be = be_b.first(bs * 2 * dmap_size);

  const T* coeffs_data = coeffs.data_handle();
  const std::size_t cstride = 2 * coeffs.extent(2);

  assert(facets0.size() == facets.size());

  const U* x_ptr = x.data_handle();
  const std::int32_t gdim = x.extent(1);
  const std::int32_t* x_dofmap_ptr = x_dofmap.data_handle();
  const std::int32_t num_x_dofs_cell = x_dofmap.extent(1);

  // P0 does not change across facets in this call, so whether it is a
  // set (non-null) transform is loop-invariant -- checked once here rather
  // than on every facet.
  const bool p0_set = is_transform_set(P0);

  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    // Cells in integration domain and test function domain meshes
    std::array<std::int32_t, 2> cells{facets(f, 0, 0), facets(f, 1, 0)};
    std::array<std::int32_t, 2> cells0{facets0(f, 0, 0), facets0(f, 1, 0)};

    // Local facet indices
    std::array<std::int32_t, 2> local_facet{facets(f, 0, 1), facets(f, 1, 1)};

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

    // Get dofmaps for cells. When integrating over interfaces between
    // two domains, the test function might only be defined on one side,
    // so we check which cells exist in the test function domain.
    std::span dmap0 = cells0[0] >= 0 ? std::span(&dmap(cells0[0], 0), dmap_size)
                                     : std::span<const std::int32_t>();
    std::span dmap1 = cells0[1] >= 0 ? std::span(&dmap(cells0[1], 0), dmap_size)
                                     : std::span<const std::int32_t>();

    // Tabulate element vector
    std::ranges::fill(be, 0);
    std::array perm = perms.empty()
                          ? std::array<std::uint8_t, 2>{0, 0}
                          : std::array{perms(cells[0], local_facet[0]),
                                       perms(cells[1], local_facet[1])};
    kernel(be.data(), coeffs_data + f * cstride, constants.data(),
           cdofs_b.data(), local_facet.data(), perm.data(), nullptr);

    if (p0_set and cells0[0] >= 0)
      P0(be, cell_info0, cells0[0], 1);
    if (p0_set and cells0[1] >= 0)
    {
      std::span sub_be(be.data() + bs * dmap_size, bs * dmap_size);
      P0(sub_be, cell_info0, cells0[1], 1);
    }

    // Add element vector to global vector
    for (std::size_t i = 0; i < dmap0.size(); ++i)
    {
      std::int32_t dof = bs * dmap0[i];
      std::int32_t offset = bs * i;
      for (int k = 0; k < bs; ++k)
        b[dof + k] += be[offset + k];
    }
    for (std::size_t i = 0; i < dmap1.size(); ++i)
    {
      std::int32_t dof = bs * dmap1[i];
      std::int32_t offset = bs * (i + dmap_size);
      for (int k = 0; k < bs; ++k)
        b[dof + k] += be[offset + k];
    }
  }
}

/// Modify RHS vector to account for boundary condition such that:
///
/// b <- b - alpha * A.(x_bc - x0)
///
/// @param[in,out] b Vector to be modified.
/// @param[in] a Bilinear form that generates A.
/// @param[in] bs0 Block size for the test function dofmap, as
/// `std::integral_constant<int, BS0>` if known at compile time, or a
/// plain `int` for the runtime-determined value.
/// @param[in] bs1 Block size for the trial function dofmap, as
/// `std::integral_constant<int, BS1>` if known at compile time, or a
/// plain `int` for the runtime-determined value.
/// @param[in] constants Constants that appear in `a`.
/// @param[in] coefficients Coefficients that appear in `a`.
/// @param[in] bc_values1 Boundary condition 'values'.
/// @param[in] bc_markers1 Indices (columns of A, rows of x) to
/// which bcs belong.
/// @param[in] x0 Array used in the lifting, typically a 'current
/// solution' in a Newton method.
/// @param[in] alpha Scaling to apply.
template <dolfinx::scalar T, std::floating_point U, typename V>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void lift_bc(V&& b, const Form<T, U>& a, auto bs0, auto bs1,
             std::span<const T> constants,
             const std::map<std::pair<IntegralType, int>,
                            std::pair<std::span<const T>, int>>& coefficients,
             std::span<const T> bc_values1,
             std::span<const std::int8_t> bc_markers1, std::span<const T> x0,
             T alpha)
{
  // Deduce runtime block sizes as fallback when compile-time sizes
  // not given. The block size of the dofmap and indexmap is the same
  // on all sub-topologies.
  assert(bs0 == a.function_spaces()[0]->dofmaps().front()->bs());
  assert(bs1 == a.function_spaces()[1]->dofmaps().front()->bs());

  auto lifting_fn = [bs0, bs1, alpha, &b, &bc_values1, &bc_markers1,
                     &x0](auto rows, auto cols, auto Ae)
  {
    const std::size_t nc = cols.size() * bs1;
    for (std::size_t i = 0; i < cols.size(); ++i)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t ii = cols[i] * bs1 + k;
        if (bc_markers1[ii])
        {
          const T x_bc = bc_values1[ii];
          const T _x0 = x0.empty() ? 0 : x0[ii];
          for (std::size_t j = 0; j < rows.size(); ++j)
          {
            for (int m = 0; m < bs0; ++m)
            {
              const std::int32_t jj = rows[j] * bs0 + m;
              b[jj] -= Ae[(j * bs0 + m) * nc + (i * bs1 + k)] * alpha
                       * (x_bc - _x0);
            }
          }
        }
      }
    }
  };

  // Use dolfinx::fem::impl::assemble_matrix assembler to work on the
  // vector b. With LiftingMode=true, the kernel is only called on cells
  // that have BC-constrained DOFs in the column space.
  std::shared_ptr<const mesh::Mesh<U>> mesh = a.mesh();
  assert(mesh);
  std::span x = mesh->geometry().x();
  md::mdspan<const U, md::extents<std::size_t, md::dynamic_extent, 3>> _x(
      x.data(), x.size() / 3, 3);
  impl::assemble_matrix<true>(lifting_fn, a, _x, constants, coefficients, {},
                              bc_markers1);
}

/// @brief Assemble linear form into a vector.
///
/// @param[in,out] b Array to be accumulated into. It will not be zeroed
/// before assembly.
/// @param[in] L Linear forms to assemble into b.
/// @param[in] x Mesh coordinates.
/// @param[in] constants Packed constants that appear in `L`.
/// @param[in] coefficients Packed coefficients that appear in `L`.
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_vector(
    V&& b, const Form<T, U>& L,
    md::mdspan<const U, md::extents<std::size_t, md::dynamic_extent, 3>> x,
    std::span<const T> constants,
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
    mdspan2_t x_dofmap = mesh->geometry().dofmaps().at(cell_type_idx);

    // Get dofmap data
    assert(L.function_spaces().at(0));
    auto element = L.function_spaces().at(0)->elements(cell_type_idx);
    assert(element);
    std::shared_ptr<const fem::DofMap> dofmap
        = L.function_spaces().at(0)->dofmaps().at(cell_type_idx);
    assert(dofmap);
    auto dofs = dofmap->map();
    const int bs = dofmap->bs();

    // Buffers reused across all integral kernels for this cell type,
    // sized for the worst case (interior facets, which touch two cells).
    std::vector<T> be_buffer(2 * bs * dofs.extent(1));
    std::vector<U> cdofs_buffer(2 * 3 * x_dofmap.extent(1));
    std::span be_b(be_buffer);
    std::span cdofs_b(cdofs_buffer);

    const fem::DofTransformKernel<T> auto& P0
        = element->template dof_transformation_fn<T>(doftransform::standard);

    std::span<const std::uint32_t> cell_info0;
    if (element->needs_dof_transformations() or L.needs_facet_permutations())
    {
      mesh0->topology_mutable()->create_entity_permutations();
      cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    }

    for (int i = 0; i < L.num_integrals(IntegralType::cell, 0); ++i)
    {
      auto fn = L.kernel(IntegralType::cell, i, cell_type_idx);
      assert(fn);
      std::span cells = L.domain(IntegralType::cell, i, cell_type_idx);
      std::span cells0 = L.domain_arg(IntegralType::cell, 0, i, cell_type_idx);
      auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
      assert(cells.size() * cstride == coeffs.size());
      if (bs == 1)
      {
        impl::assemble_cells(
            P0, b, x_dofmap, x, cells,
            std::tuple{dofs, std::integral_constant<int, 1>{}, cells0}, fn,
            constants, md::mdspan(coeffs.data(), cells.size(), cstride),
            cell_info0, be_b, cdofs_b);
      }
      else if (bs == 3)
      {
        impl::assemble_cells(
            P0, b, x_dofmap, x, cells,
            std::tuple{dofs, std::integral_constant<int, 3>(), cells0}, fn,
            constants, md::mdspan(coeffs.data(), cells.size(), cstride),
            cell_info0, be_b, cdofs_b);
      }
      else
      {
        impl::assemble_cells(P0, b, x_dofmap, x, cells,
                             std::tuple{dofs, bs, cells0}, fn, constants,
                             md::mdspan(coeffs.data(), cells.size(), cstride),
                             cell_info0, be_b, cdofs_b);
      }
    }

    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> facet_perms;
    if (L.needs_facet_permutations())
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

    using mdspanx2_t
        = md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>;
    using mdspanx22_t
        = md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2, 2>>;
    using mdspanx2x_t
        = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                          md::dynamic_extent>>;

    for (int i = 0; i < L.num_integrals(IntegralType::interior_facet, 0); ++i)
    {
      auto fn = L.kernel(IntegralType::interior_facet, i, 0);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      std::span facets = L.domain(IntegralType::interior_facet, i, 0);
      std::span facets1 = L.domain_arg(IntegralType::interior_facet, 0, i, 0);
      assert((facets.size() / 4) * 2 * cstride == coeffs.size());

      mdspanx22_t facets_mdspan(facets.data(), facets.size() / 4, 2, 2);
      mdspanx22_t facets1_mdspan(facets1.data(), facets1.size() / 4, 2, 2);
      if (bs == 1)
      {
        impl::assemble_interior_facets(
            P0, b, x_dofmap, x, facets_mdspan,
            std::tuple{dofs, std::integral_constant<int, 1>{}, facets1_mdspan},
            fn, constants,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            cell_info0, facet_perms, be_b, cdofs_b);
      }
      else if (bs == 3)
      {
        impl::assemble_interior_facets(
            P0, b, x_dofmap, x, facets_mdspan,
            std::tuple{dofs, std::integral_constant<int, 3>{}, facets1_mdspan},
            fn, constants,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            cell_info0, facet_perms, be_b, cdofs_b);
      }
      else
      {
        impl::assemble_interior_facets(
            P0, b, x_dofmap, x, facets_mdspan,
            std::tuple{dofs, bs, facets1_mdspan}, fn, constants,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            cell_info0, facet_perms, be_b, cdofs_b);
      }
    }

    for (auto itg_type : {fem::IntegralType::exterior_facet,
                          fem::IntegralType::vertex, fem::IntegralType::ridge})
    {
      md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms
          = (itg_type == fem::IntegralType::exterior_facet)
                ? facet_perms
                : md::mdspan<const std::uint8_t,
                             md::dextents<std::size_t, 2>>{};
      for (int i = 0; i < L.num_integrals(itg_type, 0); ++i)
      {
        auto fn = L.kernel(itg_type, i, 0);
        assert(fn);
        auto& [coeffs, cstride] = coefficients.at({itg_type, i});
        std::span e = L.domain(itg_type, i, 0);
        mdspanx2_t entities(e.data(), e.size() / 2, 2);
        std::span e1 = L.domain_arg(itg_type, 0, i, 0);
        mdspanx2_t entities1(e1.data(), e1.size() / 2, 2);
        assert((entities.size() / 2) * cstride == coeffs.size());
        if (bs == 1)
        {
          impl::assemble_entities(
              P0, b, x_dofmap, x, entities,
              std::tuple{dofs, std::integral_constant<int, 1>{}, entities1}, fn,
              constants, md::mdspan(coeffs.data(), entities.extent(0), cstride),
              cell_info0, perms, be_b, cdofs_b);
        }
        else if (bs == 3)
        {
          impl::assemble_entities(
              P0, b, x_dofmap, x, entities,
              std::tuple{dofs, std::integral_constant<int, 3>{}, entities1}, fn,
              constants, md::mdspan(coeffs.data(), entities.extent(0), cstride),
              cell_info0, perms, be_b, cdofs_b);
        }
        else
        {
          impl::assemble_entities(
              P0, b, x_dofmap, x, entities, std::tuple{dofs, bs, entities1}, fn,
              constants, md::mdspan(coeffs.data(), entities.extent(0), cstride),
              cell_info0, perms, be_b, cdofs_b);
        }
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
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_vector(
    V&& b, const Form<T, U>& L, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  using mdspanx3_t
      = md::mdspan<const U, md::extents<std::size_t, md::dynamic_extent, 3>>;

  std::shared_ptr<const mesh::Mesh<U>> mesh = L.mesh();
  assert(mesh);
  auto x = mesh->geometry().x();
  impl::assemble_vector(b, L, mdspanx3_t(x.data(), x.size() / 3, 3), constants,
                        coefficients);
}
} // namespace dolfinx::fem::impl
