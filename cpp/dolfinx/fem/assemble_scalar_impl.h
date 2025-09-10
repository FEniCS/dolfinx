// Copyright (C) 2019-2025 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "Form.h"
#include "FunctionSpace.h"
#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <vector>

namespace dolfinx::fem::impl
{
/// Assemble functional over cells
template <dolfinx::scalar T>
T assemble_cells(mdspan2_t x_dofmap,
                 md::mdspan<const scalar_value_t<T>,
                            md::extents<std::size_t, md::dynamic_extent, 3>>
                     x,
                 std::span<const std::int32_t> cells, FEkernel<T> auto fn,
                 std::span<const T> constants,
                 md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
                 std::span<scalar_value_t<T>> cdofs_b)
{
  T value(0);
  if (cells.empty())
    return value;

  assert(cdofs_b.size() >= 3 * x_dofmap.extent(1));

  // Iterate over all cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs_b.begin(), 3 * i));

    fn(&value, &coeffs(index, 0), constants.data(), cdofs_b.data(), nullptr,
       nullptr, nullptr);
  }

  return value;
}

/// Execute kernel over exterior facets and accumulate result
template <dolfinx::scalar T>
T assemble_exterior_facets(
    mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 2>>
        facets,
    FEkernel<T> auto fn, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<scalar_value_t<T>> cdofs_b)
{
  T value(0);
  if (facets.empty())
    return value;

  assert(cdofs_b.size() >= 3 * x_dofmap.extent(1));

  // Iterate over all facets
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    std::int32_t cell = facets(f, 0);
    std::int32_t local_facet = facets(f, 1);

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs_b.begin(), 3 * i));

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_facet);
    fn(&value, &coeffs(f, 0), constants.data(), cdofs_b.data(), &local_facet,
       &perm, nullptr);
  }

  return value;
}

/// Assemble functional over interior facets
template <dolfinx::scalar T>
T assemble_interior_facets(
    mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 2, 2>>
        facets,
    FEkernel<T> auto fn, std::span<const T> constants,
    md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                    md::dynamic_extent>>
        coeffs,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<scalar_value_t<T>> cdofs_b)
{
  T value(0);
  if (facets.empty())
    return value;

  // Create data structures used in assembly
  assert(cdofs_b.size() >= 2 * 3 * x_dofmap.extent(1));
  auto cdofs0 = cdofs_b.first(3 * x_dofmap.extent(1));
  auto cdofs1 = cdofs_b.last(3 * x_dofmap.extent(1));

  // Iterate over all facets
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    std::array cells = {facets(f, 0, 0), facets(f, 1, 0)};
    std::array local_facet = {facets(f, 0, 1), facets(f, 1, 1)};

    // Get cell geometry
    auto x_dofs0 = md::submdspan(x_dofmap, cells[0], md::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
      std::copy_n(&x(x_dofs0[i], 0), 3, std::next(cdofs0.begin(), 3 * i));
    auto x_dofs1 = md::submdspan(x_dofmap, cells[1], md::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
      std::copy_n(&x(x_dofs1[i], 0), 3, std::next(cdofs1.begin(), 3 * i));

    std::array perm = perms.empty()
                          ? std::array<std::uint8_t, 2>{0, 0}
                          : std::array{perms(cells[0], local_facet[0]),
                                       perms(cells[1], local_facet[1])};
    fn(&value, &coeffs(f, 0, 0), constants.data(), cdofs_b.data(),
       local_facet.data(), perm.data(), nullptr);
  }

  return value;
}

/// Assemble functional over vertices
template <dolfinx::scalar T>
T assemble_vertices(mdspan2_t x_dofmap,
                    md::mdspan<const scalar_value_t<T>,
                               md::extents<std::size_t, md::dynamic_extent, 3>>
                        x,
                    md::mdspan<const std::int32_t,
                               md::extents<std::size_t, md::dynamic_extent, 2>>
                        vertices,
                    FEkernel<T> auto fn, std::span<const T> constants,
                    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
                    std::span<scalar_value_t<T>> cdofs_b)
{
  T value(0);
  if (vertices.empty())
    return value;

  assert(cdofs_b.size() >= 3 * x_dofmap.extent(1));

  // Iterate over all cells
  for (std::size_t index = 0; index < vertices.extent(0); ++index)
  {
    std::int32_t cell = vertices(index, 0);
    std::int32_t local_vertex_index = vertices(index, 1);

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs_b.begin(), 3 * i));

    fn(&value, &coeffs(index, 0), constants.data(), cdofs_b.data(),
       &local_vertex_index, nullptr, nullptr);
  }

  return value;
}

/// Assemble functional into an scalar with provided mesh geometry.
template <dolfinx::scalar T, std::floating_point U>
T assemble_scalar(
    const fem::Form<T, U>& M, mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh<U>> mesh = M.mesh();
  assert(mesh);

  std::vector<scalar_value_t<T>> cdofs_b(2 * 3 * x_dofmap.extent(1));

  T value = 0;
  IntegralType cell_integral = IntegralType(0, 1);
  for (int i = 0; i < M.num_integrals(cell_integral, 0); ++i)
  {
    auto fn = M.kernel(cell_integral, i, 0);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({cell_integral, i});
    std::span<const std::int32_t> cells = M.domain(cell_integral, i, 0);
    assert(cells.size() * cstride == coeffs.size());
    value += impl::assemble_cells(
        x_dofmap, x, cells, fn, constants,
        md::mdspan(coeffs.data(), cells.size(), cstride), cdofs_b);
  }

  mesh::CellType cell_type = mesh->topology()->cell_type();
  int num_facets_per_cell
      = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
  md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms;
  if (M.needs_facet_permutations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    const std::vector<std::uint8_t>& p
        = mesh->topology()->get_facet_permutations();
    perms = md::mdspan(p.data(), p.size() / num_facets_per_cell,
                       num_facets_per_cell);
  }

  IntegralType facet_type = IntegralType(1, 1);
  for (int i = 0; i < M.num_integrals(facet_type, 0); ++i)
  {
    auto fn = M.kernel(facet_type, i, 0);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({facet_type, i});

    std::span facets = M.domain(facet_type, i, 0);

    // Two values per each adjacent cell (cell index and local facet
    // index)
    constexpr std::size_t num_adjacent_cells = 1;
    constexpr std::size_t shape1 = 2 * num_adjacent_cells;

    assert((facets.size() / 2) * cstride == coeffs.size());
    value += impl::assemble_exterior_facets(
        x_dofmap, x,
        md::mdspan<const std::int32_t,
                   md::extents<std::size_t, md::dynamic_extent, 2>>(
            facets.data(), facets.size() / shape1, 2),
        fn, constants,
        md::mdspan(coeffs.data(), facets.size() / shape1, cstride), perms,
        cdofs_b);
  }

  IntegralType interior_facet_type = IntegralType(1, 2);
  for (int i = 0; i < M.num_integrals(interior_facet_type, 0); ++i)
  {
    auto fn = M.kernel(interior_facet_type, i, 0);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({interior_facet_type, i});
    std::span facets = M.domain(interior_facet_type, i, 0);

    constexpr std::size_t num_adjacent_cells = 2;
    // Two values per each adj. cell (cell index and local facet index).
    constexpr std::size_t shape1 = 2 * num_adjacent_cells;

    assert((facets.size() / shape1) * 2 * cstride == coeffs.size());
    value += impl::assemble_interior_facets(
        x_dofmap, x,
        md::mdspan<const std::int32_t,
                   md::extents<std::size_t, md::dynamic_extent, 2, 2>>(
            facets.data(), facets.size() / shape1, 2, 2),
        fn, constants,
        md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                        md::dynamic_extent>>(
            coeffs.data(), facets.size() / shape1, 2, cstride),
        perms, cdofs_b);
  }

  std::int32_t tdim = mesh->topology()->dim();
  IntegralType vertex_type = IntegralType(tdim, 1);
  for (int i = 0; i < M.num_integrals(vertex_type, 0); ++i)
  {
    auto fn = M.kernel(vertex_type, i, 0);
    assert(fn);

    auto& [coeffs, cstride] = coefficients.at({vertex_type, i});

    std::span<const std::int32_t> vertices = M.domain(vertex_type, i, 0);
    assert(vertices.size() * cstride == coeffs.size());

    constexpr std::size_t num_adjacent_cells = 1;
    // Two values per adj. cell (cell index and local vertex index).
    constexpr std::size_t shape1 = 2 * num_adjacent_cells;

    value += impl::assemble_vertices(
        x_dofmap, x,
        md::mdspan<const std::int32_t,
                   md::extents<std::size_t, md::dynamic_extent, 2>>(
            vertices.data(), vertices.size() / shape1, shape1),
        fn, constants,
        md::mdspan(coeffs.data(), vertices.size() / shape1, cstride), cdofs_b);
  }

  return value;
}

} // namespace dolfinx::fem::impl
