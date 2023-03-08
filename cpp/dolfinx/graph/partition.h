// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <mpi.h>
#include <span>
#include <utility>
#include <vector>

#include <iostream>

namespace dolfinx::graph
{

/// @brief Signature of functions for computing the parallel
/// partitioning of a distributed graph.
// See https://github.com/doxygen/doxygen/issues/9552
/// @cond
/// @param[in] comm MPI Communicator that the graph is distributed
/// across
/// @param[in] nparts Number of partitions to divide graph nodes into
/// @param[in] local_graph Node connectivity graph
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution
/// @return Destination rank for each input node
/// @endcond
using partition_fn = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const AdjacencyList<std::int64_t>&, bool)>;

/// Partition graph across processes using the default graph partitioner
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across
/// @param[in] nparts Number of partitions to divide graph nodes into
/// @param[in] local_graph Node connectivity graph
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution
/// @return Destination rank for each input node
AdjacencyList<std::int32_t>
partition_graph(MPI_Comm comm, int nparts,
                const AdjacencyList<std::int64_t>& local_graph, bool ghosting);

/// Tools for distributed graphs
///
/// @todo Add a function that sends data to the 'owner'
namespace build
{
/// @brief Distribute adjacency list nodes to destination ranks.
///
/// The global index of each node is assumed to be the local index plus
/// the offset for this rank.
///
/// @param[in] comm MPI Communicator
/// @param[in] list The adjacency list to distribute
/// @param[in] destinations Destination ranks for the ith node in the
/// adjacency list. The first rank is the 'owner' of the node.
/// @return
/// 1. Received adjacency list for this process
/// 2. Source ranks for each node in the adjacency list
/// 3. Original global index for each node in the adjacency list
/// 4. Owner rank of ghost nodes
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
distribute(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& list,
           const graph::AdjacencyList<std::int32_t>& destinations);

/// @brief Take a set of distributed input global indices, including
/// ghosts, and determine the new global indices after remapping.
///
/// Each rank receive 'input' global indices `[i0, i1, ..., i(m-1), im,
/// ..., i(n-1)]`, where the first `m` indices are owned by the caller
/// and the remainder are 'ghosts' indices that are owned by other ranks.
///
/// Each rank assigns new global indices to its owned indices. The new
/// index is the rank offset (scan of the number of indices owned by the
/// lower rank processes, typically computed using `MPI_Exscan` with
/// `MPI_SUM`), i.e. `i1 -> offset + 1`, `i2 -> offset + 2`, etc. Ghost
/// indices are number by the remote owning processes. The function
/// returns the new ghost global indices by retrieving the new indices
/// from the owning ranks.
///
/// @param[in] comm MPI communicator
/// @param[in] owned_indices List of owned global indices. It should not
/// contain duplicates, and these indices must not appear in
/// `owned_indices` on other ranks.
/// @param[in] ghost_indices List of ghost global indices.
/// @param[in] ghost_owners The owning rank for each entry in
/// `ghost_indices`.
/// @return New global indices for the ghost indices.
std::vector<std::int64_t>
compute_ghost_indices(MPI_Comm comm,
                      std::span<const std::int64_t> owned_indices,
                      std::span<const std::int64_t> ghost_indices,
                      std::span<const int> ghost_owners);

/// Given an adjacency list with global, possibly non-contiguous, link
/// indices and a local adjacency list with contiguous link indices
/// starting from zero, compute a local-to-global map for the links.
/// Both adjacency lists must have the same shape.
///
/// @param[in] global Adjacency list with global link indices
/// @param[in] local Adjacency list with local, contiguous link indices
/// @return Map from local index to global index, which if applied to
/// the local adjacency list indices would yield the global adjacency
/// list
std::vector<std::int64_t>
compute_local_to_global_links(const graph::AdjacencyList<std::int64_t>& global,
                              const graph::AdjacencyList<std::int32_t>& local);

/// @brief Compute a local0-to-local1 map from two local-to-global maps
/// with common global indices.
///
/// @param[in] local0_to_global Map from local0 indices to global
/// indices
/// @param[in] local1_to_global Map from local1 indices to global
/// indices
/// @return Map from local0 indices to local1 indices
std::vector<std::int32_t>
compute_local_to_local(std::span<const std::int64_t> local0_to_global,
                       std::span<const std::int64_t> local1_to_global);
} // namespace build

} // namespace dolfinx::graph
