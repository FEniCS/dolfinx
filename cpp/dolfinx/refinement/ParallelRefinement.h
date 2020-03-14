// Copyright (C) 2012-2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <map>
#include <set>
#include <vector>
#include <Eigen/Dense>

namespace dolfinx
{

namespace mesh
{
class Mesh;
template <typename T>
class MeshFunction;
} // namespace mesh

namespace refinement
{
/// Data structure and methods for refining meshes in parallel

/// ParallelRefinement encapsulates two main features: a distributed
/// MeshFunction defined over the mesh edges, which can be updated
/// across processes, and storage for local mesh data, which can be used
/// to construct the new Mesh

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
  void mark(const mesh::MeshFunction<int>& refinement_marker);

  /// Transfer marked edges between processes
  void update_logical_edgefunction();

  /// Add new vertex for each marked edge, and create
  /// new_vertex_coordinates and global_edge->new_vertex mapping.
  /// Communicate new vertices with MPI to all affected processes.
  void create_new_vertices();

  /// Mapping of old edge (to be removed) to new global vertex number.
  /// Useful for forming new topology
  const std::map<std::int32_t, std::int64_t>& edge_to_new_vertex() const;

  /// Add new cells with vertex indices
  /// @param[in] idx
  void new_cells(const std::vector<std::int64_t>& idx);

  /// Use vertex and topology data to partition new mesh across processes
  /// @param[in] redistribute
  /// @return New mesh
  mesh::Mesh partition(bool redistribute) const;

  /// Build local mesh from internal data when not running in parallel
  /// @return A Mesh
  mesh::Mesh build_local() const;

private:
  // mesh::Mesh reference
  const mesh::Mesh& _mesh;

  // Mapping from old local edge index to new global vertex, needed to
  // create new topology
  std::map<std::int32_t, std::int64_t> _local_edge_to_new_vertex;

  // New storage for all coordinates when creating new vertices
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> _new_vertex_coordinates;

  // New storage for all cells when creating new topology
  std::vector<std::int64_t> _new_cell_topology;

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
