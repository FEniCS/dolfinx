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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-09-22
// Last changed: 2008-11-13

#ifndef __DYNAMIC_MESH_EDITOR_H
#define __DYNAMIC_MESH_EDITOR_H

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
    void open(Mesh& mesh, CellType::Type type, std::size_t tdim,
              std::size_t gdim, std::size_t num_global_vertices,
              std::size_t num_global_cells);

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, std::string type, std::size_t tdim, std::size_t gdim,
              std::size_t num_global_vertices, std::size_t num_global_cells);

    /// Add vertex v at given point p
    void add_vertex(std::size_t v, const Point& p);

    /// Add vertex v at given coordinate x
    void add_vertex(std::size_t v, double x);

    /// Add vertex v at given coordinate (x, y)
    void add_vertex(std::size_t v, double x, double y);

    /// Add vertex v at given coordinate (x, y, z)
    void add_vertex(std::size_t v, double x, double y, double z);

    /// Add cell with given vertices
    void add_cell(std::size_t c, const std::vector<std::size_t>& v);

    /// Add cell (interval) with given vertices
    void add_cell(std::size_t c, std::size_t v0, std::size_t v1);

    /// Add cell (triangle) with given vertices
    void add_cell(std::size_t c,  std::size_t v0, std::size_t v1,
                  std::size_t v2);

    /// Add cell (tetrahedron) with given vertices
    void add_cell(std::size_t c, std::size_t v0, std::size_t v1,
                  std::size_t v2, std::size_t v3);

    /// Close mesh, finish editing, and order entities locally
    void close(bool order=false);

  private:

    // Clear data
    void clear();

    // The mesh
    Mesh* _mesh;

    // Topological dimension
    std::size_t _tdim;

    // Geometrical (Euclidean) dimension
    std::size_t _gdim;

    // Cell type
    CellType* _cell_type;

    // Number of global vertices and cells
    std::size_t _num_global_vertices, _num_global_cells;

    // Dynamic storage for vertex coordinates
    std::vector<double> vertex_coordinates;

    // Dynamic storage for cells
    std::vector<std::size_t> cell_vertices;

  };

}

#endif
