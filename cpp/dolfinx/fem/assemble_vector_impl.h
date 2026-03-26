// Copyright (C) 2018-2025 Garth N. Wells and Paul T. Kühner
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
/// @cond
using mdspan2_t = md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>;
/// @endcond

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
template <int _bs = -1, typename V,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>

void assemble_cells(
    fem::DofTransformKernel<T> auto P0, V&& b, mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
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
  std::vector<scalar_value_t<T>> cdofs(3 * x_dofmap.extent(1));
  std::vector<T> be(bs * dmap.extent(1));

  // Iterate over active cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Integration domain celland test function cell
    std::int32_t c = cells[index];
    std::int32_t c0 = cells0[index];

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs.begin(), 3 * i));

    // Tabulate vector for cell
    std::ranges::fill(be, 0);
    kernel(be.data(), &coeffs(index, 0), constants.data(), cdofs.data(),
           nullptr, nullptr, nullptr);
    P0(be, cell_info0, c0, 1);

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
template <int _bs = -1, typename V,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_entities(
    fem::DofTransformKernel<T> auto P0, V&& b, mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2>>
        entities,
    std::tuple<mdspan2_t, int,
               md::mdspan<const std::int32_t,
                          std::extents<std::size_t, md::dynamic_extent, 2>>>
        dofmap,
    FEkernel<T> auto kernel, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const std::uint32_t> cell_info0,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms)
{
  if (entities.empty())
    return;

  const auto [dmap, bs, entities0] = dofmap;
  assert(_bs < 0 or _bs == bs);

  // Create data structures used in assembly
  const int num_dofs = dmap.extent(1);
  std::vector<scalar_value_t<T>> cdofs(3 * x_dofmap.extent(1));
  std::vector<T> be(bs * num_dofs);
  assert(entities0.size() == entities.size());
  for (std::size_t f = 0; f < entities.extent(0); ++f)
  {
    // Cell in the integration domain, local facet index relative to the
    // integration domain cell, and cell in the test function mesh
    std::int32_t cell = entities(f, 0);
    std::int32_t local_entity = entities(f, 1);
    std::int32_t cell0 = entities0(f, 0);

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs.begin(), 3 * i));

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_entity);

    // Tabulate element vector
    std::ranges::fill(be, 0);
    kernel(be.data(), &coeffs(f, 0), constants.data(), cdofs.data(),
           &local_entity, &perm, nullptr);
    P0(be, cell_info0, cell0, 1);

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
template <int _bs = -1, typename V,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_interior_facets(
    fem::DofTransformKernel<T> auto P0, V&& b, mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2, 2>>
        facets,
    std::tuple<const DofMap&, int,
               md::mdspan<const std::int32_t,
                          std::extents<std::size_t, md::dynamic_extent, 2, 2>>>
        dofmap,
    FEkernel<T> auto kernel, std::span<const T> constants,
    md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                    md::dynamic_extent>>
        coeffs,
    std::span<const std::uint32_t> cell_info0,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms)
{
  using X = scalar_value_t<T>;

  if (facets.empty())
    return;

  const auto [dmap, bs, facets0] = dofmap;
  assert(_bs < 0 or _bs == bs);

  // Create data structures used in assembly
  std::vector<X> cdofs(2 * x_dofmap.extent(1) * 3);
  std::span<X> cdofs0(cdofs.data(), x_dofmap.extent(1) * 3);
  std::span<X> cdofs1(cdofs.data() + x_dofmap.extent(1) * 3,
                      x_dofmap.extent(1) * 3);

  const std::size_t dmap_size = dmap.map().extent(1);
  std::vector<T> be(bs * 2 * dmap_size);

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
      std::copy_n(&x(x_dofs0[i], 0), 3, std::next(cdofs0.begin(), 3 * i));
    auto x_dofs1 = md::submdspan(x_dofmap, cells[1], md::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
      std::copy_n(&x(x_dofs1[i], 0), 3, std::next(cdofs1.begin(), 3 * i));

    // Get dofmaps for cells. When integrating over interfaces between
    // two domains, the test function might only be defined on one side,
    // so we check which cells exist in the test function domain.
    std::span dmap0 = cells0[0] >= 0 ? dmap.cell_dofs(cells0[0])
                                     : std::span<const std::int32_t>();
    std::span dmap1 = cells0[1] >= 0 ? dmap.cell_dofs(cells0[1])
                                     : std::span<const std::int32_t>();

    // Tabulate element vector
    std::ranges::fill(be, 0);
    std::array perm = perms.empty()
                          ? std::array<std::uint8_t, 2>{0, 0}
                          : std::array{perms(cells[0], local_facet[0]),
                                       perms(cells[1], local_facet[1])};
    kernel(be.data(), &coeffs(f, 0, 0), constants.data(), cdofs.data(),
           local_facet.data(), perm.data(), nullptr);

    if (cells0[0] >= 0)
      P0(be, cell_info0, cells0[0], 1);
    if (cells0[1] >= 0)
    {
      std::span sub_be(be.data() + bs * dmap_size, bs * dmap_size);
      P0(sub_be, cell_info0, cells0[1], 1);
    }

    // Add element vector to global vector
    if constexpr (_bs > 0)
    {
      for (std::size_t i = 0; i < dmap0.size(); ++i)
        for (int k = 0; k < _bs; ++k)
          b[_bs * dmap0[i] + k] += be[_bs * i + k];
      for (std::size_t i = 0; i < dmap1.size(); ++i)
        for (int k = 0; k < _bs; ++k)
          b[_bs * dmap1[i] + k] += be[_bs * (i + dmap_size) + k];
    }
    else
    {
      for (std::size_t i = 0; i < dmap0.size(); ++i)
        for (int k = 0; k < bs; ++k)
          b[bs * dmap0[i] + k] += be[bs * i + k];
      for (std::size_t i = 0; i < dmap1.size(); ++i)
        for (int k = 0; k < bs; ++k)
          b[bs * dmap1[i] + k] += be[bs * (i + dmap_size) + k];
    }
  }
}

/// Modify RHS vector to account for boundary condition such that:
///
/// b <- b - alpha * A.(x_bc - x0)
///
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form that generates A
/// @param[in] constants Constants that appear in `a`
/// @param[in] coefficients Coefficients that appear in `a`
/// @param[in] bc_values1 The boundary condition 'values'
/// @param[in] bc_markers1 The indices (columns of A, rows of x) to
/// which bcs belong
/// @param[in] x0 The array used in the lifting, typically a 'current
/// solution' in a Newton method
/// @param[in] alpha Scaling to apply
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void lift_bc(V&& b, const Form<T, U>& a, std::span<const T> constants,
             const std::map<std::pair<IntegralType, int>,
                            std::pair<std::span<const T>, int>>& coefficients,
             std::span<const T> bc_values1,
             std::span<const std::int8_t> bc_markers1, std::span<const T> x0,
             T alpha)
{

  // Get dofmap for columns and rows of a
  assert(a.function_spaces().at(0));
  assert(a.function_spaces().at(1));
  const int bs0 = a.function_spaces()[0]->dofmap()->bs();
  const int bs1 = a.function_spaces()[1]->dofmap()->bs();

  spdlog::debug("lifting: bs0={}, bs1={}", bs0, bs1);

  // Iterate over Form with plugin kernel

  if (bs0 > 1 or bs1 > 1)
  {
    // Lifting function for non-unit block size
    auto lifting_fn
        = [&b, &x0, &bs0, &bs1, &bc_markers1, &bc_values1,
           &alpha](std::span<const std::int32_t> rows,
                   std::span<const std::int32_t> cols, std::span<const T> Ae)
    {
      std::size_t nc = cols.size() * bs1;
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

    // Call matrix assembler in BCMode, only executing kernel on cells with BCs
    // in bc_markers1.
    assemble_matrix<T, U, true>(lifting_fn, a, constants, coefficients, {},
                                bc_markers1);
  }
  else
  {
    // Specialised lifting function for scalar values (bs0=1, bs1=1)
    auto lifting_fn
        = [&b, &x0, &bc_markers1, &bc_values1,
           &alpha](std::span<const std::int32_t> rows,
                   std::span<const std::int32_t> cols, std::span<const T> Ae)
    {
      std::size_t nr = rows.size();
      std::size_t nc = cols.size();
      for (std::size_t i = 0; i < nc; ++i)
      {
        const std::int32_t ii = cols[i];
        if (bc_markers1[ii])
        {
          const T x_bc = bc_values1[ii];
          const T _x0 = x0.empty() ? 0 : x0[ii];
          for (std::size_t j = 0; j < nr; ++j)
            b[rows[j]] -= Ae[j * nc + i] * alpha * (x_bc - _x0);
        }
      }
    };

    // Use matrix assembler in BCMode to lift RHS.
    assemble_matrix<T, U, true>(lifting_fn, a, constants, coefficients, {},
                                bc_markers1);
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
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void apply_lifting(
    V&& b,
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

      std::span<const T> _x0;
      if (!x0.empty())
        _x0 = x0[j];

      lift_bc(b, a[j]->get(), constants[j], coeffs[j],
              std::span<const T>(bc_values1), bc_markers1, _x0, alpha);
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
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_vector(
    V&& b, const Form<T, U>& L,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
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
        impl::assemble_cells<1>(
            P0, b, x_dofmap, x, cells, {dofs, bs, cells0}, fn, constants,
            md::mdspan(coeffs.data(), cells.size(), cstride), cell_info0);
      }
      else if (bs == 3)
      {
        impl::assemble_cells<3>(
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

    for (int i = 0; i < L.num_integrals(IntegralType::interior_facet, 0); ++i)
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
        impl::assemble_interior_facets<1>(
            P0, b, x_dofmap, x,
            mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
            {*dofmap, bs,
             mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
            fn, constants,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            cell_info0, facet_perms);
      }
      else if (bs == 3)
      {
        impl::assemble_interior_facets<3>(
            P0, b, x_dofmap, x,
            mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
            {*dofmap, bs,
             mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
            fn, constants,
            mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride),
            cell_info0, facet_perms);
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
            cell_info0, facet_perms);
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
          impl::assemble_entities<1>(
              P0, b, x_dofmap, x, entities, {dofs, bs, entities1}, fn,
              constants, md::mdspan(coeffs.data(), entities.extent(0), cstride),
              cell_info0, perms);
        }
        else if (bs == 3)
        {
          impl::assemble_entities<3>(
              P0, b, x_dofmap, x, entities, {dofs, bs, entities1}, fn,
              constants,
              md::mdspan(coeffs.data(), entities.size() / 2, cstride),
              cell_info0, perms);
        }
        else
        {
          impl::assemble_entities(
              P0, b, x_dofmap, x, entities, {dofs, bs, entities1}, fn,
              constants,
              md::mdspan(coeffs.data(), entities.size() / 2, cstride),
              cell_info0, perms);
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
      = md::mdspan<const scalar_value_t<T>,
                   md::extents<std::size_t, md::dynamic_extent, 3>>;

  std::shared_ptr<const mesh::Mesh<U>> mesh = L.mesh();
  assert(mesh);
  auto x = mesh->geometry().x();
  if constexpr (std::is_same_v<U, scalar_value_t<T>>)
  {
    impl::assemble_vector(b, L, mdspanx3_t(x.data(), x.size() / 3, 3),
                          constants, coefficients);
  }
  else
  {
    std::vector<scalar_value_t<T>> _x(x.begin(), x.end());
    impl::assemble_vector(b, L, mdspanx3_t(_x.data(), _x.size() / 3, 3),
                          constants, coefficients);
  }
}
} // namespace dolfinx::fem::impl
