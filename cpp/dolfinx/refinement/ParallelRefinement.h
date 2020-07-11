// Copyright (C) 2012-2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <map>
#include <set>
#include <vector>

namespace dolfinx
{

namespace mesh
{
class Mesh;
template <typename T>
class MeshTags;
} // namespace mesh

namespace common
{
class IndexMap;
}

namespace refinement
{
/// Data structure and methods for refining meshes in parallel

/// ParallelRefinement encapsulates two main features: a distributed
/// MeshTags defined over the mesh edges, which can be updated across
/// processes, and storage for local mesh data, which can be used to
/// construct the new Mesh

class ParallelRefinement
{
public:
  /// Constructor
  ParallelRefinement(const mesh::Mesh& mesh);

  /// Disable copy constructor
  ParallelRefinement(const ParallelRefinement& p) = delete;

  /// Disable copy assignment
  ParallelRefinement& operator=(const ParallelRefinement& p) = delete;

  /// Destructor
  ~ParallelRefinement();

  /// Return markers for all edges
  /// @returns array of markers
  std::vector<bool>& marked_edges();

  /// Mark edge by index
  /// @param[in] edge_index Index of edge to mark
  /// @return false if marker was already set, otherwise true
  bool mark(std::int32_t edge_index, const common::IndexMap& map_e);

  /// Mark all edges incident on entities indicated by refinement marker
  /// @param[in] refinement_marker Value 1 means "refine", any other
  ///   value means "do not refine"
  void mark(const mesh::MeshTags<std::int8_t>& refinement_marker);

  /// Transfer marked edges between processes
  static void update_logical_edgefunction(
      const MPI_Comm& neighbour_comm,
      const std::vector<std::vector<std::int32_t>>& marked_for_update,
      std::vector<bool>& marked_edges, const common::IndexMap& map_e);

  /// Add new vertex for each marked edge, and create
  /// new_vertex_coordinates and global_edge->new_vertex map.
  /// Communicate new vertices with MPI to all affected processes.
  /// @return edge_to_new_vertex map
  std::pair<
      std::map<std::int32_t, std::int64_t>,
      Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>> static create_new_vertices(const MPI_Comm&
                                                           neighbour_comm,
                                                       const std::map<
                                                           std::int32_t,
                                                           std::set<
                                                               std::int32_t>>&
                                                           shared_edges,
                                                       const mesh::Mesh& mesh,
                                                       const std::vector<bool>&
                                                           marked_edges);

  /// Use vertex and topology data to partition new mesh across
  /// processes
  /// @param[in] cell_topology Topology of cells, (vertex indices)
  /// @param[in] num_ghost_cells Number of cells which are ghost (at end
  ///   of list)
  /// @param[in] redistribute Call graph partitioner if true
  /// @return New mesh
  mesh::Mesh static partition(
      const mesh::Mesh& old_mesh,
      const std::vector<std::int64_t>& cell_topology, int num_ghost_cells,
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& new_vertex_coordinates,
      bool redistribute);

  /// Build local mesh from internal data when not running in parallel
  /// @param[in] cell_topology
  /// @return A Mesh
  mesh::Mesh static build_local(
      const mesh::Mesh& old_mesh,
      const std::vector<std::int64_t>& cell_topology,
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& new_vertex_coordinates);

  /// Adjust indices to account for extra n values on each process This
  /// is a utility to help add new topological vertices on each process
  /// into the space of the index map.
  ///
  /// @param index_map IndexMap of current mesh vertices
  /// @param n Number of new entries to be accommodated on this process
  /// @return Global indices as if "n" extra values are appended on each
  ///   process
  static std::vector<std::int64_t>
  adjust_indices(const std::shared_ptr<const common::IndexMap>& index_map,
                 std::int32_t n);

  MPI_Comm& neighbour_comm() { return _neighbour_comm; }

  std::map<std::int32_t, std::set<std::int32_t>>& shared_edges()
  {
    return _shared_edges;
  }

private:
  // Management of marked edges
  std::vector<bool> _marked_edges;

  // Temporary storage for edges that have been recently marked (global
  // index)
  std::vector<std::vector<std::int64_t>> _marked_for_update;

  // Shared edges between processes
  std::map<std::int32_t, std::set<std::int32_t>> _shared_edges;

  // Neighbourhood communicator
  MPI_Comm _neighbour_comm;
};
} // namespace refinement
} // namespace dolfinx
