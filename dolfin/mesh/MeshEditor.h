// Copyright (C) 2006-2012 Anders Logg
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
// First added:  2006-05-16
// Last changed: 2012-09-27

#ifndef __MESH_EDITOR_H
#define __MESH_EDITOR_H

#include <vector>
#include "CellType.h"

namespace dolfin
{

  class CellType;
  class Mesh;
  class Point;

  /// A simple mesh editor for creating simplicial meshes in 1D, 2D
  /// and 3D.

  class MeshEditor
  {
  public:

    /// Constructor
    MeshEditor();

    /// Destructor
    ~MeshEditor();

    /// Open mesh of given topological and geometrical dimension
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to open.
    ///     tdim (std::size_t)
    ///         The topological dimension.
    ///     gdim (std::size_t)
    ///         The geometrical dimension.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Mesh mesh;
    ///         MeshEditor editor;
    ///         editor.open(mesh, 2, 2);
    ///
    void open(Mesh& mesh, std::size_t tdim, std::size_t gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to open.
    ///     type (CellType::Type)
    ///         Cell type.
    ///     tdim (std::size_t)
    ///         The topological dimension.
    ///     gdim (std::size_t)
    ///         The geometrical dimension.
    void open(Mesh& mesh, CellType::Type type, std::size_t tdim, std::size_t gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to open.
    ///     type (std::string)
    ///         Cell type.
    ///     tdim (std::size_t)
    ///         The topological dimension.
    ///     gdim (std::size_t)
    ///         The geometrical dimension.
    void open(Mesh& mesh, std::string type, std::size_t tdim, std::size_t gdim);

    /// Specify number of vertices
    ///
    /// *Arguments*
    ///     num_vertices (std::size_t)
    ///         The number of vertices.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Mesh mesh;
    ///         MeshEditor editor;
    ///         editor.open(mesh, 2, 2);
    ///         editor.init_vertices(4);
    ///
    void init_vertices(std::size_t num_vertices);

    /// Specify number of cells
    ///
    /// *Arguments*
    ///     num_cells (std::size_t)
    ///         The number of cells.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Mesh mesh;
    ///         MeshEditor editor;
    ///         editor.open(mesh, 2, 2);
    ///         editor.init_cells(2);
    ///
    void init_cells(std::size_t num_cells);

    /// Add vertex v at given point p
    ///
    /// *Arguments*
    ///     index (std::size_t)
    ///         The vertex (index).
    ///     p (_Point_)
    ///         The point.
    void add_vertex(std::size_t index, const Point& p);

    /// Add vertex v at given coordinate x
    ///
    /// *Arguments*
    ///     index (std::size_t)
    ///         The vertex (index).
    ///     x (std::vector<double>)
    ///         The x-coordinates.
    void add_vertex(std::size_t index, const std::vector<double>& x);

    /// Add vertex v at given point x (for a 1D mesh)
    ///
    /// *Arguments*
    ///     index (std::size_t)
    ///         The vertex (index).
    ///     x (double)
    ///         The x-coordinate.
    void add_vertex(std::size_t index, double x);

    /// Add vertex v at given point (x, y) (for a 2D mesh)
    ///
    /// *Arguments*
    ///     index (std::size_t)
    ///         The vertex (index).
    ///     x (double)
    ///         The x-coordinate.
    ///     y (double)
    ///         The y-coordinate.
    void add_vertex(std::size_t index, double x, double y);

    /// Add vertex v at given point (x, y, z) (for a 3D mesh)
    ///
    /// *Arguments*
    ///     index (std::size_t)
    ///         The vertex (index).
    ///     x (double)
    ///         The x-coordinate.
    ///     y (double)
    ///         The y-coordinate.
    ///     z (double)
    ///         The z-coordinate.
    void add_vertex(std::size_t index, double x, double y, double z);

    /// Add vertex v at given point p
    ///
    /// *Arguments*
    ///     local_index (std::size_t)
    ///         The vertex (local index).
    ///     global_index (std::size_t)
    ///         The vertex (global_index).
    ///     p (_Point_)
    ///         The point.
    void add_vertex_global(std::size_t local_index, std::size_t global_index,
                           const Point& p);

    /// Add vertex v at given coordinate x
    ///
    /// *Arguments*
    ///     local_index (std::size_t)
    ///         The vertex (local index).
    ///     global_index (std::size_t)
    ///         The vertex (global_index).
    ///     x (std::vector<double>)
    ///         The x-coordinates.
    void add_vertex_global(std::size_t local_index, std::size_t global_index,
                           const std::vector<double>& x);

    /// Add cell with given vertices (1D)
    ///
    /// *Arguments*
    ///     c (std::size_t)
    ///         The cell (index).
    ///     v0 (std::vector<std::size_t>)
    ///         The first vertex (local index).
    ///     v1 (std::vector<std::size_t>)
    ///         The second vertex (local index).
    void add_cell(std::size_t c, std::size_t v0, std::size_t v1);

    /// Add cell with given vertices (2D)
    ///
    /// *Arguments*
    ///     c (std::size_t)
    ///         The cell (index).
    ///     v0 (std::vector<std::size_t>)
    ///         The first vertex (local index).
    ///     v1 (std::vector<std::size_t>)
    ///         The second vertex (local index).
    ///     v2 (std::vector<std::size_t>)
    ///         The third vertex (local index).
    void add_cell(std::size_t c, std::size_t v0, std::size_t v1, std::size_t v2);

    /// Add cell with given vertices (3D)
    ///
    /// *Arguments*
    ///     c (std::size_t)
    ///         The cell (index).
    ///     v0 (std::vector<std::size_t>)
    ///         The first vertex (local index).
    ///     v1 (std::vector<std::size_t>)
    ///         The second vertex (local index).
    ///     v2 (std::vector<std::size_t>)
    ///         The third vertex (local index).
    ///     v3 (std::vector<std::size_t>)
    ///         The fourth vertex (local index).
    void add_cell(std::size_t c, std::size_t v0, std::size_t v1,
                  std::size_t v2, std::size_t v3);

    /// Add cell with given vertices
    ///
    /// *Arguments*
    ///     c (std::size_t)
    ///         The cell (index).
    ///     v (std::vector<std::size_t>)
    ///         The vertex indices (local indices)
    void add_cell(std::size_t c, const std::vector<std::size_t>& v);

    /// Add cell with given vertices
    ///
    /// *Arguments*
    ///     local_index (std::size_t)
    ///         The cell (index).
    ///     global_index (std::size_t)
    ///         The global (user) cell index.
    ///     v (std::vector<std::size_t>)
    ///         The vertex indices (local indices)
    void add_cell(std::size_t local_index, std::size_t global_index,
                  const std::vector<std::size_t>& v);

    /// Close mesh, finish editing, and order entities locally
    ///
    /// *Arguments*
    ///     order (bool)
    ///         Order entities locally if true. Default values is true.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         MeshEditor editor;
    ///         editor.open(mesh, 2, 2);
    ///         ...
    ///         editor.close()
    ///
    void close(bool order=true);

  private:

    // Friends
    friend class TetrahedronCell;

    // Add vertex, common part
    void add_vertex_common(std::size_t v, std::size_t dim);

    // Add cell, common part
    void add_cell_common(std::size_t v, std::size_t dim);

    // Compute boundary indicators (exterior facets)
    void compute_boundary_indicators();

    // Clear all data
    void clear();

    // Check that vertices are in range
    void check_vertices(const std::vector<std::size_t>& v) const;

    // The mesh
    Mesh* _mesh;

    // Topological dimension
    std::size_t _tdim;

    // Geometrical (Euclidean) dimension
    std::size_t _gdim;

    // Number of vertices
    std::size_t _num_vertices;

    // Number of cells
    std::size_t _num_cells;

    // Next available vertex
    std::size_t next_vertex;

    // Next available cell
    std::size_t next_cell;

    // Temporary storage for local cell data
    std::vector<std::size_t> _vertices;

  };

}

#endif
