// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <map>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace mesh
{

class Topology;

class PartitioningNew
{
public:
  /// Compute markers for interior/boundary vertices
  /// @param[in] topology_local Local topology
  /// @return Array where the ith entry is true if the ith vertex is on
  /// the boundary
  static std::vector<bool>
  compute_vertex_exterior_markers(const mesh::Topology& topology_local);

  /// @todo Return list of neighbour processes
  ///
  /// Compute new, contiguous global indices from a collection of
  /// global, possibly globally non-contiguous, indices and assign
  /// process ownership to the new global indices such that the global
  /// index of owned indices increases with increasing MPI rank.
  ///
  /// @param[in] comm The communicator across which the indices are
  ///   distributed
  /// @param[in] global_indices Global induces on this process
  /// @param[in] shared_indices Vector that is true for indices on the
  ///   exterior locally
  /// @return {Local (old, from local_to_global) -> local (new) indices,
  ///   global indices for ghosts of this process}. The new indices are
  ///   [0, ..., N), with [0, ..., n0) being owned. The new global index
  ///   for an owned indices is n_global = n + offset, where offset is
  ///   computed from a process scan. Indices [n0, ..., N) are owned by
  ///   a remote process and the ghosts return vector maps [n0, ..., N)
  ///   to global indices.
  static std::pair<std::vector<std::int32_t>, std::vector<std::int64_t>>
  reorder_global_indices(MPI_Comm comm,
                         const std::vector<std::int64_t>& global_indices,
                         const std::vector<bool>& shared_indices);

  /// Compute destination rank for mesh cells using a graph partitioner
  /// @param[in] comm MPI Communicator
  /// @param[in] nparts Number of partitions
  /// @param[in] cell_type Cell type
  /// @param[in] cells Cells on this process. The ith entry in list
  ///   contains the global indices for the cell vertices. Each cell can
  ///   appear only once across all procsss. The cell vertex indices are
  ///   not necessarily contiguous globally, i.e. the maximum index
  ///   across all processes can be greater than the number of vertices.
  /// @return Destination process for each cell on this process
  static std::vector<int>
  partition_cells(MPI_Comm comm, int nparts, const mesh::CellType cell_type,
                  const graph::AdjacencyList<std::int64_t>& cells);

  /// Compute a local AdjacencyList list with contiguous indices from an
  /// AdjacencyList that map have non-contiguous data
  /// @param[in] list Adjacency list with links that might not have
  ///   contiguous numdering
  /// @return Adjacency list with contiguous ordering [0, 1, ..., n),
  ///   and a map from local indices in the returned Adjacency list to
  ///   the global indices in @p list
  static std::pair<graph::AdjacencyList<std::int32_t>,
                   std::vector<std::int64_t>>
  create_local_adjacency_list(const graph::AdjacencyList<std::int64_t>& list);

  /// Compute a distributed AdjacencyList list from an AdjacencyList
  /// that map have non-contiguous data
  /// @param[in] comm
  /// @param[in] topology_local
  /// @param[in] local_to_global_vertices
  static std::tuple<graph::AdjacencyList<std::int32_t>, common::IndexMap>
  create_distributed_adjacency_list(
      MPI_Comm comm, const mesh::Topology& topology_local,
      const std::vector<std::int64_t>& local_to_global_vertices);

  /// Re-distribute adjacency list across processes
  /// @param[in] comm MPI Communicator
  /// @param[in] list An adjacency list
  /// @param[in] owner Destination rank for the ith entry in the
  ///   adjacency list
  /// @return Adjacency list for this process and a vector of source
  ///   processes for entry in the adjacency list
  static std::pair<graph::AdjacencyList<std::int64_t>, std::vector<int>>
  distribute(const MPI_Comm& comm,
             const graph::AdjacencyList<std::int64_t>& list,
             const std::vector<int>& owner);
};
} // namespace mesh
} // namespace dolfinx
