// Copyright (C) 2008-2013 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <cstdint>
#include <dolfin/common/types.h>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dolfin
{
namespace mesh
{
// Developer note: MeshFunction and MeshValueCollection cannot appear in
// the implementations that appear in this file of the templated
// functions as this leads to a circular dependency. Therefore the
// functions are templated over these types.

class Mesh;
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

/// This class partitions and distributes a mesh based on partitioned
/// local mesh data.The local mesh data will also be repartitioned and
/// redistributed during the computation of the mesh partitioning.
///
/// After partitioning, each process has a local mesh and some data that
/// couples the meshes together.

class Partitioning
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
  /// @param part_processes_ratio
  ///     Percentage of active processes that will be used to partition the
  ///     mesh.
  /// @param graph_partitioner
  ///     External Graph Partitioner (SCOTCH, PARMETIS)
  static mesh::Mesh
  build_distributed_mesh(const MPI_Comm& comm, mesh::CellType::Type cell_type,
                         const Eigen::Ref<const EigenRowArrayXXd> points,
                         const Eigen::Ref<const EigenRowArrayXXi64> cells,
                         const std::vector<std::int64_t>& global_cell_indices,
                         const mesh::GhostMode ghost_mode,
                         const double part_processes_ratio = 1.0,
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
                   std::map<std::int32_t, std::set<std::int32_t>>>
  distribute_points(const MPI_Comm mpi_comm,
                    const Eigen::Ref<const EigenRowArrayXXd> points,
                    const std::vector<std::int64_t>& global_point_indices);

  // Utility to create global vertex indices, needed for higher order
  // meshes, where there are geometric points which are not at the
  // vertex nodes
  static std::pair<std::int64_t, std::vector<std::int64_t>>
  build_global_vertex_indices(
      MPI_Comm mpi_comm, std::int32_t num_vertices,
      const std::vector<std::int64_t>& global_point_indices,
      const std::map<std::int32_t, std::set<std::int32_t>>& shared_points);
};
} // namespace mesh
} // namespace dolfin
