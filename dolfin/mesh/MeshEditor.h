// Copyright (C) 2006-2009 Anders Logg
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
// Last changed: 2011-07-18

#ifndef __MESH_EDITOR_H
#define __MESH_EDITOR_H

#include <vector>
#include <dolfin/common/types.h>
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
    ///     tdim (uint)
    ///         The topological dimension.
    ///     gdim (uint)
    ///         The geometrical dimension.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Mesh mesh;
    ///         MeshEditor editor;
    ///         editor.open(mesh, 2, 2);
    ///
    void open(Mesh& mesh, uint tdim, uint gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to open.
    ///     type (CellType::Type)
    ///         Cell type.
    ///     tdim (uint)
    ///         The topological dimension.
    ///     gdim (uint)
    ///         The geometrical dimension.
    void open(Mesh& mesh, CellType::Type type, uint tdim, uint gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to open.
    ///     type (std::string)
    ///         Cell type.
    ///     tdim (uint)
    ///         The topological dimension.
    ///     gdim (uint)
    ///         The geometrical dimension.
    void open(Mesh& mesh, std::string type, uint tdim, uint gdim);

    /// Specify number of vertices
    ///
    /// *Arguments*
    ///     num_vertices (uint)
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
    void init_vertices(uint num_vertices);

    /// Specify number of cells
    ///
    /// *Arguments*
    ///     num_cells (uint)
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
    void init_cells(uint num_cells);

    /// Add vertex v at given point p
    ///
    /// *Arguments*
    ///     v (uint)
    ///         The vertex (index).
    ///     p (_Point_)
    ///         The point.
    void add_vertex(uint v, const Point& p);

    /// Add vertex v at given coordinate x
    ///
    /// *Arguments*
    ///     v (uint)
    ///         The vertex (index).
    ///     x (std::vector<double>)
    ///         The x-coordinates.
    void add_vertex(uint v, const std::vector<double>& x);

    /// Add cell with given vertices
    ///
    /// *Arguments*
    ///     c (uint)
    ///         The cell (index).
    ///     v (std::vector<uint>)
    ///         The vertex indices
    void add_cell(uint c, const std::vector<uint>& v);

    /// Add cell (interval) with given vertices
    ///
    /// *Arguments*
    ///     c (uint)
    ///         The cell (index).
    ///     v0 (uint)
    ///         Index of the first vertex.
    ///     v1 (uint)
    ///         Index of the second vertex.
    void add_cell(uint c, uint v0, uint v1);

    /// Add cell (triangle) with given vertices
    ///
    /// *Arguments*
    ///     c (uint)
    ///         The cell (index).
    ///     v0 (uint)
    ///         Index of the first vertex.
    ///     v1 (uint)
    ///         Index of the second vertex.
    ///     v2 (uint)
    ///         Index of the third vertex.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         MeshEditor editor;
    ///         editor.add_cell(0, 0, 1, 2);
    ///
    void add_cell(uint c, uint v0, uint v1, uint v2);

    /// Add cell (tetrahedron) with given vertices
    ///
    /// *Arguments*
    ///     c (uint)
    ///         The cell (index).
    ///     v0 (uint)
    ///         Index of the first vertex.
    ///     v1 (uint)
    ///         Index of the second vertex.
    ///     v2 (uint)
    ///         Index of the third vertex.
    ///     v3 (uint)
    ///         Index of the fourth vertex.
    void add_cell(uint c, uint v0, uint v1, uint v2, uint v3);

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
    void add_vertex_common(uint v, uint dim);

    // Add cell, common part
    void add_cell_common(uint v, uint dim);

    // Compute boundary indicators (exterior facets)
    void compute_boundary_indicators();

    // Clear all data
    void clear();

    // Check that vertex is in range
    void check_vertex(uint v);

    // The mesh
    Mesh* mesh;

    // Topological dimension
    uint tdim;

    // Geometrical (Euclidean) dimension
    uint gdim;

    // Number of vertices
    uint num_vertices;

    // Number of cells
    uint num_cells;

    // Next available vertex
    uint next_vertex;

    // Next available cell
    uint next_cell;

    // Temporary storage for local cell data
    std::vector<uint> vertices;

  };

}

#endif
