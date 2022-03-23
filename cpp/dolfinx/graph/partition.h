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
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

#include <iostream>

namespace dolfinx::graph
{

/// Signature of functions for computing the parallel partitioning of a
/// distributed graph
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across
/// @param[in] nparts Number of partitions to divide graph nodes into
/// @param[in] local_graph Node connectivity graph
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution
/// @return Destination rank for each input node
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

/// @todo Make sense of this docstring
///
/// Compute ghost indices in a global IndexMap space, from a list of
/// arbitrary global indices, where the ghosts are at the end of the
/// list, and their owning processes are known.
/// @param[in] comm MPI communicator
/// @param[in] owned_indices List of arbitrary global indices, with
/// ghosts at end
/// @param[in] ghost_indices List of arbitrary global indices, with
/// ghosts at end
/// @param[in] ghost_owners List of owning process for each ghost index
/// @return Indexing of ghosts in a global space starting from 0 on process 0
std::vector<std::int64_t>
compute_ghost_indices(MPI_Comm comm,
                      const xtl::span<const std::int64_t>& owned_indices,
                      const xtl::span<const std::int64_t>& ghost_indices,
                      const xtl::span<const int>& ghost_owners);

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

/// Compute a local0-to-local1 map from two local-to-global maps with
/// common global indices
///
/// @param[in] local0_to_global Map from local0 indices to global
/// indices
/// @param[in] local1_to_global Map from local1 indices to global
/// indices
/// @return Map from local0 indices to local1 indices
std::vector<std::int32_t>
compute_local_to_local(const xtl::span<const std::int64_t>& local0_to_global,
                       const xtl::span<const std::int64_t>& local1_to_global);
} // namespace build

} // namespace dolfinx::graph
