// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mpi.h>
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
/// computing entity-to-vertex connectivity `(dim, 0)`, and
/// cell-to-entity connectivity `(tdim, dim)`.
///
/// Computed entities are oriented such that their local (to the
/// process) orientation agrees with their global orientation
///
/// @param[in] comm MPI Communicator
/// @param[in] topology Mesh topology
/// @param[in] dim The dimension of the entities to create
/// @param[in] index Index of entity in dimension `dim` as listed in
/// `Topology::entity_types(dim)`.
/// @return Tuple of (cell-entity connectivity, entity-vertex
/// connectivity, index map, list of interprocess entities).
/// Interprocess entities lie on the "true" boundary between owned cells
/// of each process. If the entities already exists, then {nullptr,
/// nullptr, nullptr, std::vector()} is returned.
std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>, std::vector<std::int32_t>>
compute_entities(MPI_Comm comm, const Topology& topology, int dim, int index);

/// @brief Compute connectivity (d0 -> d1) for given pair of entity
/// types, given by topological dimension and index, as found in
/// `Topology::entity_types()`
/// @param[in] topology The topology
/// @param[in] d0 The dimension and index of the entities
/// @param[in] d1 The dimension and index of the incident entities
/// @returns The connectivities [(d0 -> d1), (d1 -> d0)] if they are
/// computed. If (d0, d1) already exists then a nullptr is returned. If
/// (d0, d1) is computed and the computation of (d1, d0) was required as
/// part of computing (d0, d1), the (d1, d0) is returned as the second
/// entry. The second entry is otherwise nullptr.
std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
compute_connectivity(const Topology& topology,
                     std::pair<std::int8_t, std::int8_t> d0,
                     std::pair<std::int8_t, std::int8_t> d1);

} // namespace dolfinx::mesh
