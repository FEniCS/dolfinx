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
#include <optional>
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
                 std::span<scalar_value_t<T>> cdofs_b,
                 std::optional<void*> custom_data = std::nullopt)
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
       nullptr, custom_data.value_or(nullptr));
  }

  return value;
}

/// @brief Execute kernel over entities of codimension ≥ 1 and accumulate result
/// in a scalar.
///
/// Each entity is represented by (i) a cell that the entity is attached to
/// and (ii) the local index of the entity  with respect to the cell. The
/// kernel is executed for each entity. The kernel can access data
/// (e.g., coefficients, basis functions) associated with the attached cell.
/// However, entities may be attached to more than one cell. This function
/// therefore computes 'one-sided' integrals, i.e. evaluates integrals as seen
/// from cell used to define the entity.
/// @param custom_data Optional pointer to user-supplied data passed to the
/// kernel at runtime.
template <dolfinx::scalar T>
T assemble_entities(
    mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 2>>
        entities,
    FEkernel<T> auto fn, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<scalar_value_t<T>> cdofs_b,
    std::optional<void*> custom_data = std::nullopt)
{
  T value(0);
  if (entities.empty())
    return value;

  assert(cdofs_b.size() >= 3 * x_dofmap.extent(1));

  // Iterate over all facets
  for (std::size_t f = 0; f < entities.extent(0); ++f)
  {
    std::int32_t cell = entities(f, 0);
    std::int32_t local_entity = entities(f, 1);

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs_b.begin(), 3 * i));

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_entity);
    fn(&value, &coeffs(f, 0), constants.data(), cdofs_b.data(), &local_entity,
       &perm, custom_data.value_or(nullptr));
  }

  return value;
}

/// Assemble functional over interior facets
/// @param custom_data Optional pointer to user-supplied data passed to the
/// kernel at runtime.
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
    std::span<scalar_value_t<T>> cdofs_b,
    std::optional<void*> custom_data = std::nullopt)
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
       local_facet.data(), perm.data(), custom_data.value_or(nullptr));
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
  for (int i = 0; i < M.num_integrals(IntegralType::cell, 0); ++i)
  {
    auto fn = M.kernel(IntegralType::cell, i, 0);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::optional<void*> custom_data = M.custom_data(IntegralType::cell, i, 0);
    std::span<const std::int32_t> cells = M.domain(IntegralType::cell, i, 0);
    assert(cells.size() * cstride == coeffs.size());
    value += impl::assemble_cells(
        x_dofmap, x, cells, fn, constants,
        md::mdspan(coeffs.data(), cells.size(), cstride), cdofs_b, custom_data);
  }

  mesh::CellType cell_type = mesh->topology()->cell_type();
  int num_facets_per_cell
      = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
  md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> facet_perms;
  if (M.needs_facet_permutations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    const std::vector<std::uint8_t>& p
        = mesh->topology()->get_facet_permutations();
    facet_perms = md::mdspan(p.data(), p.size() / num_facets_per_cell,
                             num_facets_per_cell);
  }

  for (int i = 0; i < M.num_integrals(IntegralType::interior_facet, 0); ++i)
  {
    auto fn = M.kernel(IntegralType::interior_facet, i, 0);
    assert(fn);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::interior_facet, i});
    std::optional<void*> custom_data
        = M.custom_data(IntegralType::interior_facet, i, 0);
    std::span facets = M.domain(IntegralType::interior_facet, i, 0);

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
        facet_perms, cdofs_b, custom_data);
  }

  for (auto itg_type : {fem::IntegralType::exterior_facet,
                        fem::IntegralType::vertex, fem::IntegralType::ridge})
  {
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms
        = (itg_type == fem::IntegralType::exterior_facet)
              ? facet_perms
              : md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>>{};

    for (int i = 0; i < M.num_integrals(itg_type, 0); ++i)
    {
      auto fn = M.kernel(itg_type, i, 0);
      assert(fn);
      auto& [coeffs, cstride] = coefficients.at({itg_type, i});
      std::optional<void*> custom_data = M.custom_data(itg_type, i, 0);

      std::span entities = M.domain(itg_type, i, 0);

      // Two values per each adj. cell (cell index and local entity index).
      assert((entities.size() / 2) * cstride == coeffs.size());
      value += impl::assemble_entities(
          x_dofmap, x,
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>(
              entities.data(), entities.size() / 2, 2),
          fn, constants,
          md::mdspan(coeffs.data(), entities.size() / 2, cstride), perms,
          cdofs_b, custom_data);
    }
  }

  return value;
}

} // namespace dolfinx::fem::impl
