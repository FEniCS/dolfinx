// Copyright (C) 2008-2013 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DistributedMeshTools.h"
#include "Mesh.h"
#include "PartitionData.h"
#include <cstdint>
#include <dolfin/common/Set.h>
#include <dolfin/common/types.h>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dolfin
{
namespace mesh
{
// Developer note: MeshFunction and MeshValueCollection cannot
// appear in the implementations that appear in this file of the
// templated functions as this leads to a circular
// dependency. Therefore the functions are templated over these
// types.

template <typename T>
class MeshFunction;
template <typename T>
class MeshValueCollection;
class CellType;

/// Enum for different partitioning ghost modes
enum class GhostMode : int
{
  none,
  shared_facet,
  shared_vertex
};

/// This class partitions and distributes a mesh based on
/// partitioned local mesh data.The local mesh data will also be
/// repartitioned and redistributed during the computation of the
/// mesh partitioning.
///
/// After partitioning, each process has a local mesh and some data
/// that couples the meshes together.

class MeshPartitioning
{
public:
  /// Build distributed mesh from a set of points and cells on each local
  /// process
  /// @param comm
  ///     MPI Communicator
  /// @param type
  ///     Cell type
  /// @param points
  ///     Geometric points on each process, numbered from process 0 upwards.
  /// @param cells
  ///     Topological cells with global vertex indexing. Each cell appears once
  ///     only.
  /// @param global_cell_indices
  ///     Global index for each cell
  /// @param ghost_mode
  ///     Ghost mode
  static mesh::Mesh
  build_distributed_mesh(const MPI_Comm& comm, mesh::CellType::Type type,
                         const Eigen::Ref<const EigenRowArrayXXd>& points,
                         const Eigen::Ref<const EigenRowArrayXXi64>& cells,
                         const std::vector<std::int64_t>& global_cell_indices,
                         const mesh::GhostMode ghost_mode,
                         std::string graph_partitioner = "SCOTCH");

  /// Redistribute points to the processes that need them.
  /// @param mpi_comm
  ///   MPI Communicator
  /// @param points
  ///   Existing vertex coordinates array on each process before
  ///   distribution
  /// @param global_point_indices
  ///   Global indices for vertices required on this process
  /// @return
  ///   vertex_coordinates (array of coordinates on this process after
  ///   distribution) and shared_vertices_local (map from local index to set of
  ///   sharing processes for each shared vertex)
  static std::pair<EigenRowArrayXXd,
                   std::map<std::int32_t, std::set<std::uint32_t>>>
  distribute_points(const MPI_Comm mpi_comm,
                    const Eigen::Ref<const EigenRowArrayXXd>& points,
                    const std::vector<std::int64_t>& global_point_indices);

  /// Compute mapping of globally indexed vertices to local indices
  /// and remap topology accordingly
  ///
  /// @param mpi_comm
  ///   MPI Communicator
  /// @param cell_vertices
  ///   Input cell topology (global indexing)
  /// @param cell_permutation
  ///   Permutation from VTK to DOLFIN index ordering
  /// @return
  ///   Local-to-global map for vertices (std::vector<std::int64_t>) and cell
  ///   topology in local indexing (EigenRowArrayXXi32)
  static std::tuple<std::uint64_t, std::vector<std::int64_t>,
                    EigenRowArrayXXi32>
  compute_point_mapping(std::uint32_t num_cell_vertices,
                        const Eigen::Ref<const EigenRowArrayXXi64>& cell_points,
                        const std::vector<std::uint8_t>& cell_permutation);

  // Utility to create global vertex indices, needed for higher
  // order meshes, where there are geometric points which are not
  // at the vertex nodes
  static std::pair<std::int64_t, std::vector<std::int64_t>>
  build_global_vertex_indices(
      MPI_Comm mpi_comm, std::uint32_t num_vertices,
      const std::vector<std::int64_t>& global_point_indices,
      const std::map<std::int32_t, std::set<std::uint32_t>>& shared_points);

private:
  // Compute cell partitioning from local mesh data. Returns a
  // vector 'cell -> process' vector for cells, and
  // a map 'local cell index -> processes' to which ghost cells must
  // be sent
  static PartitionData
  partition_cells(const MPI_Comm& mpi_comm, mesh::CellType::Type cell_type,
                  const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
                  const std::string partitioner);

  // Build a distributed mesh from local mesh data with a computed
  // partition
  static mesh::Mesh build(const MPI_Comm& comm, mesh::CellType::Type type,
                          const Eigen::Ref<const EigenRowArrayXXi64>& cells,
                          const Eigen::Ref<const EigenRowArrayXXd>& points,
                          const std::vector<std::int64_t>& global_cell_indices,
                          const mesh::GhostMode ghost_mode,
                          const PartitionData& mp);

  // Distribute additional cells implied by connectivity via vertex. The input
  // cell_vertices, shared_cells, global_cell_indices and cell_partition must
  // already be distributed with a ghost layer by shared_facet.
  // FIXME: shared_cells, cell_vertices, global_cell_indices and cell_partition
  // are all modified by this function.
  static void distribute_cell_layer(
      MPI_Comm mpi_comm, const int num_regular_cells,
      std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
      EigenRowArrayXXi64& cell_vertices,
      std::vector<std::int64_t>& global_cell_indices,
      std::vector<int>& cell_partition);

  // FIXME: make clearer what goes in and what comes out
  // Reorder cells by Gibbs-Poole-Stockmeyer algorithm (via SCOTCH). Returns
  // the tuple (reordered_shared_cells, reordered_cell_vertices,
  // reordered_global_cell_indices)
  static std::tuple<std::map<std::int32_t, std::set<std::uint32_t>>,
                    EigenRowArrayXXi64, std::vector<std::int64_t>>
  reorder_cells_gps(
      MPI_Comm mpi_comm, const std::uint32_t num_regular_cells,
      const mesh::CellType& cell_type,
      const std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
      const Eigen::Ref<const EigenRowArrayXXi64>& global_cell_vertices,
      const std::vector<std::int64_t>& global_cell_indices);

  // FIXME: Update, making clear exactly what is computed
  // This function takes the partition computed by the partitioner
  // (which tells us to which process each of the local cells stored on
  //  this process belongs) and sends the cells
  // to the appropriate owning process. Ghost cells are also sent,
  // along with the list of sharing processes.
  // Returns (new_cell_vertices, new_global_cell_indices,
  // new_cell_partition, shared_cells, number of non-ghost cells on this
  // process).
  static std::tuple<
      EigenRowArrayXXi64, std::vector<std::int64_t>, std::vector<int>,
      std::map<std::int32_t, std::set<std::uint32_t>>, std::int32_t>
  distribute_cells(const MPI_Comm mpi_comm,
                   const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
                   const std::vector<std::int64_t>& global_cell_indices,
                   const PartitionData& mp);

  // FIXME: Improve explanation
  // Utility to convert received_point_indices into
  // point sharing information
  static std::map<std::int32_t, std::set<std::uint32_t>> build_shared_points(
      MPI_Comm mpi_comm,
      const std::vector<std::vector<std::size_t>>& received_point_indices,
      const std::pair<std::size_t, std::size_t> local_point_range,
      const std::vector<std::vector<std::uint32_t>>& local_indexing);
};
} // namespace mesh
} // namespace dolfin
