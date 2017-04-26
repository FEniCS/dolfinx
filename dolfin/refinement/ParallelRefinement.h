// Copyright (C) 2012-2014 Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
//
// First Added: 2013-01-02

#ifndef __PARALLEL_REFINEMENT_H
#define __PARALLEL_REFINEMENT_H

#include <unordered_map>
#include <vector>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template<typename T> class EdgeFunction;
  template<typename T> class MeshFunction;

  /// Data structure and methods for refining meshes in parallel

  /// ParallelRefinement encapsulates two main features:
  /// a distributed EdgeFunction, which can be updated
  /// across processes, and storage for local mesh data,
  /// which can be used to construct the new Mesh

  class ParallelRefinement
  {
  public:

    /// Constructor
    ParallelRefinement(const Mesh& mesh);

    /// Destructor
    ~ParallelRefinement();

    /// Original mesh associated with this refinement
    const Mesh& mesh() const
    { return _mesh; }

    /// Return marked status of edge
    /// @param edge_index (std::size_t)
    bool is_marked(std::size_t edge_index) const;

    /// Mark edge by index
    /// @param edge_index (std::size_t)
    ///  Index of edge to mark
    void mark(std::size_t edge_index);

    /// Mark all edges in mesh
    void mark_all();

    /// Mark all edges incident on entities indicated by refinement
    /// marker
    /// @param refinement_marker (const MeshFunction<bool>)
    void mark(const MeshFunction<bool>& refinement_marker);

    /// Mark all incident edges of an entity
    /// @param cell (MeshEntity)
    void mark(const MeshEntity& cell);

    /// Return list of marked edges incident on this MeshEntity -
    /// usually a cell
    /// @param cell (const _MeshEntity_)
    std::vector<std::size_t> marked_edge_list(const MeshEntity& cell) const;

    /// Transfer marked edges between processes
    void update_logical_edgefunction();

    /// Add new vertex for each marked edge, and create
    /// new_vertex_coordinates and global_edge->new_vertex mapping.
    /// Communicate new vertices with MPI to all affected processes.
    void create_new_vertices();

    /// Mapping of old edge (to be removed) to new global vertex
    /// number. Useful for forming new topology
    std::shared_ptr<const std::map<std::size_t, std::size_t> > edge_to_new_vertex() const;

    /// Add a new cell to the list in 3D or 2D
    /// @param cell (const _Cell_)
    void new_cell(const Cell& cell);

    /// Add a new cell with vertex indices
    /// @param i0 (std::size_t)
    /// @param i1 (std::size_t)
    /// @param i2 (std::size_t)
    /// @param i3 (std::size_t)
    void new_cell(std::size_t i0, std::size_t i1, std::size_t i2,
                  std::size_t i3);

    /// Add a new cell with vertex indices
    /// @param i0 (std::size_t)
    /// @param i1 (std::size_t)
    /// @param i2 (std::size_t)
    void new_cell(std::size_t i0, std::size_t i1, std::size_t i2);

    /// Add new cells with vertex indices
    /// @param idx (const std::vector<std::size_t>)
    void new_cells(const std::vector<std::size_t>& idx);

    /// Use vertex and topology data to partition new mesh across processes
    /// @param new_mesh (_Mesh_)
    /// @param redistribute (bool)
    void partition(Mesh& new_mesh, bool redistribute) const;

    /// Build local mesh from internal data when not running in parallel
    /// @param new_mesh (_Mesh_)
    void build_local(Mesh& new_mesh) const;

  private:

    // Mesh reference
    const Mesh& _mesh;

    // Shared edges between processes. In R^2, vector size is 1
    std::unordered_map<unsigned int, std::vector<std::pair<unsigned int,
      unsigned int> > > shared_edges;

    // Mapping from old local edge index to new global vertex, needed
    // to create new topology
    std::shared_ptr<std::map<std::size_t, std::size_t> > local_edge_to_new_vertex;

    // New storage for all coordinates when creating new vertices
    std::vector<double> new_vertex_coordinates;

    // New storage for all cells when creating new topology
    std::vector<std::size_t> new_cell_topology;

    // Management of marked edges
    std::vector<bool> marked_edges;

    // Temporary storage for edges that have been recently marked
    std::vector<std::vector<std::size_t> > marked_for_update;
  };

}

#endif
