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
  const std::vector<bool>& marked_edges() const;

  /// Mark edge by index
  /// @param[in] edge_index Index of edge to mark
  /// @return false if marker was already set, otherwise true
  bool mark(std::int32_t edge_index);

  /// Mark all edges in mesh
  void mark_all();

  /// Mark all edges incident on entities indicated by refinement marker
  /// @param[in] refinement_marker Value 1 means "refine", any other
  ///   value means "do not refine"
  void mark(const mesh::MeshTags<std::int8_t>& refinement_marker);

  /// Transfer marked edges between processes
  void update_logical_edgefunction();

  /// Add new vertex for each marked edge, and create
  /// new_vertex_coordinates and global_edge->new_vertex map.
  /// Communicate new vertices with MPI to all affected processes.
  /// @return edge_to_new_vertex map
  std::map<std::int32_t, std::int64_t> create_new_vertices();

  /// Use vertex and topology data to partition new mesh across
  /// processes
  /// @param[in] cell_topology Topology of cells, (vertex indices)
  /// @param[in] num_ghost_cells Number of cells which are ghost (at end
  ///   of list)
  /// @param[in] redistribute Call graph partitioner if true
  /// @return New mesh
  mesh::Mesh partition(const std::vector<std::int64_t>& cell_topology,
                       int num_ghost_cells, bool redistribute) const;

  /// Build local mesh from internal data when not running in parallel
  /// @param[in] cell_topology
  /// @return A Mesh
  mesh::Mesh build_local(const std::vector<std::int64_t>& cell_topology) const;

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

private:
  // Mesh
  const mesh::Mesh& _mesh;

  // New storage for all coordinates when creating new vertices
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _new_vertex_coordinates;

  // Management of marked edges
  std::vector<bool> _marked_edges;

  // Temporary storage for edges that have been recently marked (global
  // index)
  std::vector<std::vector<std::int64_t>> _marked_for_update;

  // Shared edges between processes
  std::map<std::int32_t, std::set<std::int32_t>> _shared_edges;

  // Neighborhood communicator
  MPI_Comm _neighbor_comm;
};
} // namespace refinement
} // namespace dolfinx
