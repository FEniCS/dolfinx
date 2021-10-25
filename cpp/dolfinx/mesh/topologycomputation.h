// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <tuple>

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

/// Compute mesh entities of given topological dimension by computing
/// entity-to-vertex connectivity (dim, 0), and cell-to-entity
/// connectivity (tdim, dim)
/// @param[in] comm MPI Communicator
/// @param[in] topology Mesh topology
/// @param[in] dim The dimension of the entities to create
/// @return Tuple of (cell-entity connectivity, entity-vertex
/// connectivity, index map). If the entities already exist, then
/// {nullptr, nullptr, nullptr} is returned.
std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>>
compute_entities(MPI_Comm comm, const Topology& topology, int dim);

/// Compute connectivity (d0 -> d1) for given pair of topological
/// dimensions
/// @param[in] topology The topology
/// @param[in] d0 The dimension of the nodes in the adjacency list
/// @param[in] d1 The dimension of the edges in the adjacency list
/// @returns The connectivities [(d0, d1), (d1, d0)] if they are
/// computed. If (d0, d1) already exists then a nullptr is returned. If
/// (d0, d1) is computed and the computation of (d1, d0) was required as
/// part of computing (d0, d1), the (d1, d0) is returned as the second
/// entry. The second entry is otherwise nullptr.
std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
compute_connectivity(const Topology& topology, int d0, int d1);

} // namespace dolfinx::mesh
