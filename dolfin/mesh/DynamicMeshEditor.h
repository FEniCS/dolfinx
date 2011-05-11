// Copyright (C) 2008 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-09-22
// Last changed: 2008-11-13

#ifndef __DYNAMIC_MESH_EDITOR_H
#define __DYNAMIC_MESH_EDITOR_H

#include <dolfin/common/types.h>
#include "CellType.h"

namespace dolfin
{

  class Mesh;
  class Point;
  class Vector;

  /// This class provides an interface for dynamic editing of meshes,
  /// that is, when the number of vertices and cells are not known
  /// a priori. If the number of vertices and cells are known a priori,
  /// it is more efficient to use the default editor MeshEditor.

  class DynamicMeshEditor
  {
  public:

    /// Constructor
    DynamicMeshEditor();

    /// Destructor
    ~DynamicMeshEditor();

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, CellType::Type type, uint tdim, uint gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, std::string type, uint tdim, uint gdim);

    /// Add vertex v at given point p
    void add_vertex(uint v, const Point& p);

    /// Add vertex v at given coordinate x
    void add_vertex(uint v, double x);

    /// Add vertex v at given coordinate (x, y)
    void add_vertex(uint v, double x, double y);

    /// Add vertex v at given coordinate (x, y, z)
    void add_vertex(uint v, double x, double y, double z);

    /// Add cell with given vertices
    void add_cell(uint c, const std::vector<uint>& v);

    /// Add cell (interval) with given vertices
    void add_cell(uint c, uint v0, uint v1);

    /// Add cell (triangle) with given vertices
    void add_cell(uint c, uint v0, uint v1, uint v2);

    /// Add cell (tetrahedron) with given vertices
    void add_cell(uint c, uint v0, uint v1, uint v2, uint v3);

    /// Close mesh, finish editing, and order entities locally
    void close(bool order=false);

  private:

    // Clear data
    void clear();

    // The mesh
    Mesh* mesh;

    // Topological dimension
    uint tdim;

    // Geometrical (Euclidean) dimension
    uint gdim;

    // Cell type
    CellType* cell_type;

    // Dynamic storage for vertex coordinates
    std::vector<double> vertex_coordinates;

    // Dynamic storage for cells
    std::vector<uint> cell_vertices;

  };

}

#endif
