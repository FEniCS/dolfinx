// Copyright (C) 2008-2020 Niclas Jansson, Ola Skavhaug, Anders Logg, Garth N.
// Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "PartitionData.h"
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/graph/CSRGraph.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace graph
{
template <typename T>
class AdjacencyList;
}

namespace mesh
{

/// Enum for different partitioning ghost modes
enum class GhostMode : int
{
  none,
  shared_facet,
  shared_vertex
};

/// Enum for different external graph partitioners
enum class Partitioner
{
  scotch,
  parmetis,
  kahip
};

/// This class partitions and distributes a mesh based on partitioned
/// local mesh data.The local mesh data will also be repartitioned and
/// redistributed during the computation of the mesh partitioning.
///
/// After partitioning, each process has a local mesh and some data that
/// couples the meshes together.

class Partitioning
{
public:
  /// NEW: Compute destination rank for mesh cells using a graph partitioner
  /// @param[in] comm MPI Communicator
  /// @param[in] nparts Number of partitions
  /// @param[in] cell_type Cell type
  /// @param[in] cells Cells on this process. The ith entry list the
  ///   global indices for the cell vertices. Each cell can appears only
  ///   once across all procsss
  /// @return Destination process for each cell on this process
  static std::vector<int>
  partition_cells(MPI_Comm comm, int nparts, const mesh::CellType cell_type,
                  const graph::AdjacencyList<std::int64_t>& cells);

  /// Compute a local AdjacencyList list from a AdjacencyList that map
  /// have non-contiguous data
  /// @param[in] list Adjacency list with links that might not have
  ///   contiguous numdering
  /// @return Adjacency list with contiguous ordering [0, 1, ..., n), a
  ///   a map from the global ordering in the cells to the local
  ///   ordering, and the value n
  static std::tuple<graph::AdjacencyList<std::int32_t>,
                    std::map<std::int64_t, std::int32_t>, std::int32_t>
  create_local_adjacency_list(const graph::AdjacencyList<std::int64_t>& list);

  /// NEW: Re-distribute adjacency list across processes
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

  /// Build distributed mesh from a set of points and cells on each
  /// local process
  /// @param[in] comm MPI Communicator
  /// @param[in] cell_type Cell type
  /// @param[in] points Geometric points on each process, numbered from
  ///   process 0 upwards.
  /// @param[in] cells Topological cells with global vertex indexing.
  ///   Each cell appears once only.
  /// @param[in] global_cell_indices Global index for each cell
  /// @param[in] ghost_mode Ghost mode
  /// @param[in] graph_partitioner External Graph Partitioner (SCOTCH,
  ///   ParMETIS, etc)
  /// @return A distributed mesh
  static mesh::Mesh build_distributed_mesh(
      const MPI_Comm& comm, mesh::CellType cell_type,
      const Eigen::Ref<const Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& points,
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          cells,
      const std::vector<std::int64_t>& global_cell_indices,
      const mesh::GhostMode ghost_mode,
      const mesh::Partitioner graph_partitioner = mesh::Partitioner::scotch);

  /// Build distributed mesh from a set of points and cells on each
  /// local process with a pre-computed partition
  /// @param[in] comm MPI Communicator
  /// @param[in] type Cell type
  /// @param[in] points Geometric points on each process, numbered from
  ///   process 0 upwards.
  /// @param[in] cell_vertices Topological cells with global vertex
  ///   indexing. Each cell appears once only.
  /// @param[in] global_cell_indices Global index for each cell
  /// @param[in] ghost_mode Ghost mode
  /// @param[in] cell_partition Cell partition data
  /// @return A distributed mesh
  static mesh::Mesh build_from_partition(
      const MPI_Comm& comm, mesh::CellType type,
      const Eigen::Ref<const Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& points,
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          cell_vertices,
      const std::vector<std::int64_t>& global_cell_indices,
      const mesh::GhostMode ghost_mode, const PartitionData& cell_partition);

  /// Partition mesh cells using an external Graph Partitioner
  /// @param[in] comm MPI Communicator
  /// @param[in] nparts Number of partitions
  /// @param[in] cell_type Cell type
  /// @param[in] cell_vertices Topological cells with global vertex
  ///   indexing. Each cell appears once only.
  /// @param[in] graph_partitioner The graph partitioner
  /// @return Cell partition data
  static PartitionData partition_cells(
      const MPI_Comm& comm, int nparts, const mesh::CellType cell_type,
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          cell_vertices,
      const mesh::Partitioner graph_partitioner);

  /// Redistribute points to the processes that need them
  /// @param[in] comm MPI Communicator
  /// @param[in] points Existing vertex coordinates array on each
  ///   process before distribution
  /// @param[in] global_point_indices Global indices for vertices
  ///   required on this process
  /// @return vertex_coordinates (array of coordinates on this process
  ///   after distribution) and shared_points (map from global index to
  ///   set of sharing processes for each shared point)
  static std::tuple<
      std::shared_ptr<common::IndexMap>, std::vector<std::int64_t>,
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  distribute_points(
      const MPI_Comm comm,
      Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
          points,
      const std::vector<std::int64_t>& global_point_indices);

  /// Utility to compute halo cells for a given custom cell partition
  /// @param[in] comm MPI Communicator
  /// @param[in] partition Array of destination process for each local
  ///   cell
  /// @param[in] cell_type Cell type
  /// @param[in] cell_vertices Topological cells with global vertex
  ///   indexing
  /// @return ghost_procs Map of cell_index to vector of sharing
  ///   processes for those cells that have multiple owners
  static std::map<std::int64_t, std::vector<int>> compute_halo_cells(
      MPI_Comm comm, std::vector<int> partition, const mesh::CellType cell_type,
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          cell_vertices);
};
} // namespace mesh
} // namespace dolfinx
