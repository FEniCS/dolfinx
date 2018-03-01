// Copyright (C) 2008-2013 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DistributedMeshTools.h"
#include "LocalMeshValueCollection.h"
#include "Mesh.h"
#include <boost/multi_array.hpp>
#include <cstdint>
#include <dolfin/common/Set.h>
#include <dolfin/log/log.h>
#include <map>
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
class LocalMeshData;
class CellType;

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
  /// Build a distributed mesh from a local mesh on process 0
  static void build_distributed_mesh(Mesh& mesh);

  /// Build a distributed mesh from a local mesh on process 0, with
  /// distribution of cells supplied (destination processes for each
  /// cell)
  static void build_distributed_mesh(Mesh& mesh,
                                     const std::vector<int>& cell_partition,
                                     const std::string ghost_mode);

  /// Build a distributed mesh from 'local mesh data' that is
  /// distributed across processes
  static void build_distributed_mesh(Mesh& mesh, const LocalMeshData& data,
                                     const std::string ghost_mode);

private:
  // Compute cell partitioning from local mesh data. Returns a
  // vector 'cell -> process' vector for cells in LocalMeshData, and
  // a map 'local cell index -> processes' to which ghost cells must
  // be sent
  static void
  partition_cells(const MPI_Comm& mpi_comm, const LocalMeshData& mesh_data,
                  const std::string partitioner,
                  std::vector<int>& cell_partition,
                  std::map<std::int64_t, std::vector<int>>& ghost_procs);

  // Build a distributed mesh from local mesh data with a computed
  // partition
  static void build(Mesh& mesh, const LocalMeshData& data,
                    const std::vector<int>& cell_partition,
                    const std::map<std::int64_t, std::vector<int>>& ghost_procs,
                    const std::string ghost_mode);

  // FIXME: Improve this docstring
  // Distribute a layer of cells attached by vertex to boundary updating
  // new_mesh_data and shared_cells. Used when ghosting by vertex.
  static void distribute_cell_layer(
      MPI_Comm mpi_comm, const int num_regular_cells,
      const std::int64_t num_global_vertices,
      std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
      boost::multi_array<std::int64_t, 2>& cell_vertices,
      std::vector<std::int64_t>& global_cell_indices,
      std::vector<int>& cell_partition);

  // FIXME: make clearer what goes in and what comes out
  // Reorder cells by Gibbs-Poole-Stockmeyer algorithm (via SCOTCH). Returns
  // the tuple (new_shared_cells, new_cell_vertices,new_global_cell_indices).
  static void reorder_cells_gps(
      MPI_Comm mpi_comm, const std::uint32_t num_regular_cells,
      const mesh::CellType& cell_type,
      const std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
      const boost::multi_array<std::int64_t, 2>& cell_vertices,
      const std::vector<std::int64_t>& global_cell_indices,
      std::map<std::int32_t, std::set<std::uint32_t>>& reordered_shared_cells,
      boost::multi_array<std::int64_t, 2>& reordered_cell_vertices,
      std::vector<std::int64_t>& reordered_global_cell_indices);

  // FIXME: make clearer what goes in and what comes out
  // Reorder vertices by Gibbs-Poole-Stockmeyer algorithm (via SCOTCH).
  // Returns the pair (new_vertex_indices, new_vertex_global_to_local).
  static void reorder_vertices_gps(
      MPI_Comm mpi_comm, const std::int32_t num_regular_vertices,
      const std::int32_t num_regular_cells, const int num_cell_vertices,
      const boost::multi_array<std::int64_t, 2>& cell_vertices,
      const std::vector<std::int64_t>& vertex_indices,
      const std::map<std::int64_t, std::int32_t>& vertex_global_to_local,
      std::vector<std::int64_t>& reordered_vertex_indices,
      std::map<std::int64_t, std::int32_t>& reordered_vertex_global_to_local);

  // FIXME: Update, making clear exactly what is computed
  // This function takes the partition computed by the partitioner
  // (which tells us to which process each of the local cells stored in
  // LocalMeshData on this process belongs) and sends the cells
  // to the appropriate owning process. Ghost cells are also sent,
  // along with the list of sharing processes.
  // A new LocalMeshData object is populated with the redistributed
  // cells. Return the number of non-ghost cells on this process.
  static std::int32_t distribute_cells(
      const MPI_Comm mpi_comm, const LocalMeshData& data,
      const std::vector<int>& cell_partition,
      const std::map<std::int64_t, std::vector<int>>& ghost_procs,
      boost::multi_array<std::int64_t, 2>& new_cell_vertices,
      std::vector<std::int64_t>& new_global_cell_indices,
      std::vector<int>& new_cell_partition,
      std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells);

  // FIXME: Improve explaination
  // Utility to convert received_vertex_indices into
  // vertex sharing information
  static void build_shared_vertices(
      MPI_Comm mpi_comm,
      std::map<std::int32_t, std::set<std::uint32_t>>& shared_vertices,
      const std::map<std::int64_t, std::int32_t>&
          vertex_global_to_local_indices,
      const std::vector<std::vector<std::size_t>>& received_vertex_indices);

  // FIXME: make clear what is computed
  // Distribute vertices and vertex sharing information
  static void distribute_vertices(
      const MPI_Comm mpi_comm, const LocalMeshData& mesh_data,
      const std::vector<std::int64_t>& vertex_indices,
      boost::multi_array<double, 2>& new_vertex_coordinates,
      std::map<std::int64_t, std::int32_t>& vertex_global_to_local_indices,
      std::map<std::int32_t, std::set<std::uint32_t>>& shared_vertices_local);

  // Compute the local->global and global->local maps for all local vertices
  // on this process, from the global vertex indices on each local cell.
  // Returns the number of regular (non-ghosted) vertices.
  static std::int32_t compute_vertex_mapping(
      MPI_Comm mpi_comm, const std::int32_t num_regular_cells,
      const boost::multi_array<std::int64_t, 2>& cell_vertices,
      std::vector<std::int64_t>& vertex_indices,
      std::map<std::int64_t, std::int32_t>& vertex_global_to_local);

  // FIXME: Improve pre-conditions explaination
  // Build mesh
  static void build_local_mesh(
      Mesh& mesh, const std::vector<std::int64_t>& global_cell_indices,
      const boost::multi_array<std::int64_t, 2>& cell_global_vertices,
      const mesh::CellType::Type cell_type, const int tdim,
      const std::int64_t num_global_cells,
      const std::vector<std::int64_t>& vertex_indices,
      const boost::multi_array<double, 2>& vertex_coordinates, const int gdim,
      const std::int64_t num_global_vertices,
      const std::map<std::int64_t, std::int32_t>&
          vertex_global_to_local_indices);
};
}
}