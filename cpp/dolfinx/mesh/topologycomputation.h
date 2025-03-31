// Copyright (C) 2006-2024 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::graph
{
template <typename T>
class AdjacencyList;
}

namespace dolfinx::mesh
{
class Topology;

/// @brief Compute mesh entities of given topological dimension by
/// computing cell-to-entity `(tdim, i) -> `(dim, entity_type)` and
/// entity-to-vertex connectivity `(dim, entity_type) -> `(0, 0)`
/// connectivity.
///
/// Computed entities are oriented such that their local (to the
/// process) orientation agrees with their global orientation
///
/// @param[in] topology Mesh topology.
/// @param[in] dim Dimension of the entities to create.
/// @param[in] entity_type Entity type in dimension `dim` to create.
/// Entity type must be in the list returned by Topology::entity_types.
/// @return Tuple of (cell->entity connectivity, entity->vertex
/// connectivity, index map for created entities, list of interprocess
/// entities). Interprocess entities lie on the "true" boundary between
/// owned cells of each process. If entities of type `entity_type`
/// already exists, then {nullptr, nullptr, nullptr, std::vector()} is
/// returned.
std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>, std::vector<std::int32_t>>
compute_entities(const Topology& topology, int dim, CellType entity_type);

/// @brief Compute connectivity (d0 -> d1) for given pair of entity
/// types, given by topological dimension and index, as found in
/// `Topology::entity_types()`
/// @param[in] topology The topology
/// @param[in] d0 Dimension and index of the entities, `(dim0, i)`.
/// @param[in] d1 Dimension and index of the incident entities, `(dim1,
/// j)`.
/// @returns The connectivities [(d0 -> d1), (d1 -> d0)] if they are
/// computed. If (d0, d1) already exists then a nullptr is returned. If
/// (d0, d1) is computed and the computation of (d1, d0) was required as
/// part of computing (d0, d1), the (d1, d0) is returned as the second
/// entry. The second entry is otherwise nullptr.
std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
compute_connectivity(const Topology& topology, std::array<int, 2> d0,
                     std::array<int, 2> d1);

} // namespace dolfinx::mesh
