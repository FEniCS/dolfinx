// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <set>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace graph
{

/// Tools for distributed graphs
///
/// TODO: Add a function that sends data (Eigen arrays) to the 'owner'

class Partitioning
{
public:
  /// @todo Return the list of neighbour processes which is computed
  /// internally
  ///
  /// Compute new, contiguous global indices from a collection of
  /// global, possibly globally non-contiguous, indices and assign
  /// process ownership to the new global indices such that the global
  /// index of owned indices increases with increasing MPI rank.
  ///
  /// @param[in] comm The communicator across which the indices are
  ///   distributed
  /// @param[in] global_indices Global indices on this process. Some
  ///   global indices may also be on other processes
  /// @param[in] shared_indices Vector that is true for indices that may
  ///   also be in other process. Size is the same as @p global_indices.
  /// @return {Local (old, from local_to_global) -> local (new) indices,
  ///   global indices for ghosts of this process}. The new indices are
  ///   [0, ..., N), with [0, ..., n0) being owned. The new global index
  ///   for an owned index is n_global = n + offset, where offset is
  ///   computed from a process scan. Indices [n0, ..., N) are owned by
  ///   a remote process and the ghosts return vector maps [n0, ..., N)
  ///   to global indices.
  static std::pair<std::vector<std::int32_t>, std::vector<std::int64_t>>
  reorder_global_indices(MPI_Comm comm,
                         const std::vector<std::int64_t>& global_indices,
                         const std::vector<bool>& shared_indices);

  /// Compute a local AdjacencyList list with contiguous indices from an
  /// AdjacencyList that may have non-contiguous data
  ///
  /// @param[in] list Adjacency list with links that might not have
  ///   contiguous numdering
  /// @return Adjacency list with contiguous ordering [0, 1, ..., n),
  ///   and a map from local indices in the returned Adjacency list to
  ///   the global indices in @p list
  static std::pair<graph::AdjacencyList<std::int32_t>,
                   std::vector<std::int64_t>>
  create_local_adjacency_list(const graph::AdjacencyList<std::int64_t>& list);

  /// Build a distributed AdjacencyList list with re-numbered links from
  /// an AdjacencyList that may have non-contiguous data. The
  /// distribution of the AdjacencyList nodes is unchanged.
  ///
  /// @param[in] comm MPI communicator
  /// @param[in] list_local Local adjacency list, with contiguous link
  ///   indices
  /// @param[in] local_to_global_links Local-to-global map for links in
  ///   the local adjacency list
  /// @param[in] shared_links Try for possible shared links
  static std::tuple<graph::AdjacencyList<std::int32_t>, common::IndexMap>
  create_distributed_adjacency_list(
      MPI_Comm comm, const graph::AdjacencyList<std::int32_t>& list_local,
      const std::vector<std::int64_t>& local_to_global_links,
      const std::vector<bool>& shared_links);

  /// Distribute adjacency list nodes to destination ranks. The global
  /// index of each node is assumed to be the local index plus the
  /// offset for this rank.
  ///
  /// @param[in] comm MPI Communicator
  /// @param[in] list The adjacency list to distribute
  /// @param[in] destinations Destination ranks for the ith node in the
  ///   adjacency list
  /// @return Adjacency list for this process, array of source ranks for
  ///   each node in the adjacency list, and the original global index
  ///   for each node.
  static std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
                    std::vector<std::int64_t>, std::vector<int>>
  distribute(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& list,
             const graph::AdjacencyList<std::int32_t>& destinations);

  /// Compute ghost indices in a global IndexMap space, from a list of arbitrary
  /// global indices, where the ghosts are at the end of the list, and their
  /// owning processes are known.
  /// @param[in] comm MPI communicator
  /// @param[in] global_indices List of arbitrary global indices, ghosts at end
  /// @param[in] ghost_owners List of owning processes of the ghost indices
  /// @return Indexing of ghosts in a global space starting from 0 on process 0
  static std::vector<std::int64_t>
  compute_ghost_indices(MPI_Comm comm,
                        const std::vector<std::int64_t>& global_indices,
                        const std::vector<int>& ghost_owners);

  /// Distribute data to process ranks where it it required
  ///
  /// @param[in] comm The MPI communicator
  /// @param[in] indices Global indices of the data required by this
  ///   process
  /// @param[in] x Data on this process which may be distributed (by
  ///   row). The global index for the [0, ..., n) local rows is assumed
  ///   to be the local index plus the offset for this process
  /// @return The data for each index in @p indices
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  distribute_data(
      MPI_Comm comm, const std::vector<std::int64_t>& indices,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& x);

  /// Given an adjacency list with global, possibly non-contiguous, link
  /// indices and a local adjacency list with contiguous link indices
  /// starting from zero, compute a local-to-global map for the links.
  /// Both adjacency lists must have the same shape.
  ///
  /// @param[in] global Adjacency list with global link indices
  /// @param[in] local Adjacency list with local, contiguous link
  ///   indices
  /// @return Map from local index to global index, which if applied to
  /// the local adjacency list indices would yield the global adjacency
  /// list
  static std::vector<std::int64_t> compute_local_to_global_links(
      const graph::AdjacencyList<std::int64_t>& global,
      const graph::AdjacencyList<std::int32_t>& local);

  /// Compute a local0-to-local1 map from two local-to-global maps with
  /// common global indices
  ///
  /// @param[in] local0_to_global Map from local0 indices to global
  ///   indices
  /// @param[in] local1_to_global Map from local1 indices to global
  ///   indices
  /// @return Map from local0 indices to local1 indices
  static std::vector<std::int32_t>
  compute_local_to_local(const std::vector<std::int64_t>& local0_to_global,
                         const std::vector<std::int64_t>& local1_to_global);
};
} // namespace graph
} // namespace dolfinx
