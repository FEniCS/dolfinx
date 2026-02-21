// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "AdjacencyList.h"
#include <dolfinx/common/IndexMap.h>
#include <string>
#include <tuple>
#include <utility>

namespace dolfinx::graph
{
/// @brief Compute an directed graph that describes the parallel
/// communication patterns.
///
/// The graph describes the communication pattern for a 'forward
/// scatter', i.e. sending owned data to ranks that ghost the data
/// (owner->ghost operation).
///
/// Each node in the graph corresponds to an MPI rank. A graph edge is
/// a forward (owner->ghost) communication path. The edge weight is
/// the number 'values' communicated along the edge. Each edge also
/// has a marker that indicates if the edge is sending data to:
///
/// 1. A node (rank) that shares memory with the sender (`true`), or
///
/// 2. A remote node that does not share memory with the sender
///   (`false`).
///
/// The graph data can be visualised using a tool like
/// [NetworkX](https://networkx.org/),
///
/// @note Collective.
///
/// @param[in] map Index map to build the graph for.
/// @param[in] root MPI rank on which to build the communication
/// graph data.
/// @return Adjacency list representing the communication pattern.
/// Edges data is (0) the edge, (1) edge weight (`weight`) and (2)
/// local/remote is memory indicator (`local==1` is an edge to a
/// shared memory node). Node data is (number of owned indices, number
/// of ghost indices).
AdjacencyList<std::vector<std::tuple<int, std::size_t, std::int8_t>>,
              std::vector<std::int32_t>,
              std::vector<std::pair<std::int32_t, std::int32_t>>>
comm_graph(const common::IndexMap& map, int root = 0);

/// @brief Build communication graph data as a JSON string.
///
/// The data string can be decoded (loaded) to create a Python object
/// from which a [NetworkX](https://networkx.org/) graph can be
/// constructed.
///
/// See ::comm_graph for a description of the data.
///
/// @param[in] g Communication graph.
/// @return JSON string representing the communication graph. Edge
/// data is data volume (`weight`) and local/remote memory indicator
/// (`local==1` is an edge to an shared memory process/rank, other
/// wise the target node is a remote memory rank).
std::string comm_to_json(
    const AdjacencyList<std::vector<std::tuple<int, std::size_t, std::int8_t>>,
                        std::vector<std::int32_t>,
                        std::vector<std::pair<std::int32_t, std::int32_t>>>& g);
} // namespace dolfinx::graph
