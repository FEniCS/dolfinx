// Copyright (C) 2013-2026 Johan Hake, Jan Blechta, Garth N. Wells and Paul T.
// Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Form.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>
#include <ranges>
#include <span>
#include <vector>

/// @file integration_domains.h
/// @brief Functions for computing integration domains.

namespace dolfinx::fem
{
namespace impl
{
/// Helper function to get an array of (cell, local_facet) pairs
/// corresponding to a given facet index.
/// @param[in] f Facet index
/// @param[in] cells List of cells incident to the facet
/// @param[in] c_to_f Cell to facet connectivity
/// @return Vector of (cell, local_facet) pairs
template <int num_cells>
std::array<std::int32_t, 2 * num_cells>
get_cell_facet_pairs(std::int32_t f, std::span<const std::int32_t> cells,
                     const graph::AdjacencyList<std::int32_t>& c_to_f)
{
  // Loop over cells sharing facet
  assert(cells.size() == num_cells);
  std::array<std::int32_t, 2 * num_cells> cell_local_facet_pairs;
  for (int c = 0; c < num_cells; ++c)
  {
    // Get local index of facet with respect to the cell
    std::int32_t cell = cells[c];
    auto cell_facets = c_to_f.links(cell);
    auto facet_it = std::find(cell_facets.begin(), cell_facets.end(), f);
    assert(facet_it != cell_facets.end());
    int local_f = std::ranges::distance(cell_facets.begin(), facet_it);
    cell_local_facet_pairs[2 * c] = cell;
    cell_local_facet_pairs[2 * c + 1] = local_f;
  }

  return cell_local_facet_pairs;
}

/// Helper function to get an array of of (cell, local_entity) pairs
/// corresponding to a given entity index.
/// @note If the entity is connected to multiple cells, the first one is picked.
/// @param[in] e entity index
/// @param[in] cells List of cells incident to the entity
/// @param[in] c_to_e Cell to entity connectivity
/// @return Vector of (cell, local_entity) pairs
template <int num_cells>
std::array<std::int32_t, 2 * num_cells>
get_cell_entity_pairs(std::int32_t e, std::span<const std::int32_t> cells,
                      const graph::AdjacencyList<std::int32_t>& c_to_e)
{
  static_assert(num_cells == 1); // Patch assembly not supported.

  assert(cells.size() > 0);

  // Use first cell for assembly over by default
  std::int32_t cell = cells[0];

  // Find local index of entity within cell
  auto cell_entities = c_to_e.links(cell);
  auto it = std::ranges::find(cell_entities, e);
  assert(it != cell_entities.end());
  std::int32_t local_index = std::ranges::distance(cell_entities.begin(), it);

  return {cell, local_index};
}

} // namespace impl

/// @brief Given an integral type and a set of entities, computes and
/// return data for the entities that should be integrated over.
///
/// This function returns a list data, for each entity in  `entities`,
/// that is used in assembly. For cell integrals it is simply the cell
/// cell indices. For exterior facet integrals, a list of `(cell_index,
/// local_facet_index)` pairs is returned. For interior facet integrals,
/// a list of `(cell_index0, local_facet_index0, cell_index1,
/// local_facet_index1)` tuples is returned.
/// The data computed by this function is typically used as input to
/// fem::create_form.
///
/// @note Owned mesh entities only are returned. Ghost entities are not
/// included.
///
/// @pre For facet integrals, the topology facet-to-cell and
/// cell-to-facet connectivity must be computed before calling this
/// function.
///
/// @param[in] integral_type Integral type.
/// @param[in] topology Mesh topology.
/// @param[in] entities List of mesh entities. Depending on the `IntegralType`
/// these are associated with different entities:
///     `IntegralType::cell`:             cells
///     `IntegralType::exterior_facet`: facets
///     `IntegralType::interior_facet`:   facets
///     `IntegralType::vertex`:           vertices
/// @return List of integration entity data, depending on the `IntegralType` the
/// data per entity has different layouts
///     `IntegralType::cell`:             cell
///     `IntegralType::exterior_facet`:   (cell, local_facet)
///     `IntegralType::interior_facet`:   (cell, local_facet)
///     `IntegralType::vertex`:           (cell, local_vertex)
std::vector<std::int32_t>
compute_integration_domains(IntegralType integral_type,
                            const mesh::Topology& topology,
                            std::span<const std::int32_t> entities);
} // namespace dolfinx::fem
