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

namespace dolfinx
{

namespace mesh
{
class Mesh;
class MeshEntity;
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

  /// Destructor
  ~ParallelRefinement() = default;

  /// Original mesh associated with this refinement
  const mesh::Mesh& mesh() const;

  /// Return marked status of edge
  /// @param[in] edge_index
  bool is_marked(std::int32_t edge_index) const;

  /// Mark edge by index
  /// @param[in] edge_index Index of edge to mark
  void mark(std::int32_t edge_index);

  /// Mark all edges in mesh
  void mark_all();

  /// Mark all edges incident on entities indicated by refinement marker
  /// @param[in] refinement_marker Value 1 means "refine", any other
  ///   value means "do not refine"
  void mark(const mesh::MeshFunction<int>& refinement_marker);

  /// Mark all incident edges of an entity
  /// @param[in] cell
  void mark(const mesh::MeshEntity& cell);

  /// Return list of marked edges incident on this mesh::MeshEntity -
  /// usually a cell
  /// @param[in] cell
  std::vector<std::size_t> marked_edge_list(const mesh::MeshEntity& cell) const;

  /// Transfer marked edges between processes
  void update_logical_edgefunction();

  /// Add new vertex for each marked edge, and create
  /// new_vertex_coordinates and global_edge->new_vertex mapping.
  /// Communicate new vertices with MPI to all affected processes.
  void create_new_vertices();

  /// Mapping of old edge (to be removed) to new global vertex number.
  /// Useful for forming new topology
  const std::map<std::size_t, std::size_t>& edge_to_new_vertex() const;

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
  std::map<std::size_t, std::size_t> _local_edge_to_new_vertex;

  // New storage for all coordinates when creating new vertices
  std::vector<double> _new_vertex_coordinates;

  // New storage for all cells when creating new topology
  std::vector<std::int64_t> _new_cell_topology;

  // Management of marked edges
  std::vector<bool> _marked_edges;

  // Temporary storage for edges that have been recently marked (global
  // index)
  std::vector<std::vector<std::int64_t>> _marked_for_update;

  // Shared edges between processes
  std::map<std::int32_t, std::set<std::int32_t>> _shared_edges;

  // Mapping from global to local index (only for shared edges)
  std::map<std::int64_t, std::int32_t> _global_to_local_edge_map;
};
} // namespace refinement
} // namespace dolfinx
