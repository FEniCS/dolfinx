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
// Last changed: 2011-02-09

#ifndef __MESH_EDITOR_H
#define __MESH_EDITOR_H

#include <vector>
#include <dolfin/common/types.h>
#include "CellType.h"

namespace dolfin
{

  class Mesh;
  class Point;

  /// A simple mesh editor for creating simplicial meshes in 1D, 2D and 3D.

  class MeshEditor
  {
  public:

    /// Constructor
    MeshEditor();

    /// Destructor
    ~MeshEditor();

    /// Open mesh of given topological and geometrical dimension
    void open(Mesh& mesh, uint tdim, uint gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, CellType::Type type, uint tdim, uint gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, std::string type, uint tdim, uint gdim);

    /// Specify number of vertices
    void init_vertices(uint num_vertices);

    /// Specify number of vertices
    void init_higher_order_vertices(uint num_higher_order_vertices);

    /// Specify number of cells
    void init_cells(uint num_cells);

    /// Specify number of cells
    void init_higher_order_cells(uint num_higher_order_cells, uint num_higher_order_cell_dof);

    /// Set boolean indicator inside MeshGeometry
    void set_affine_cell_indicator(uint c, const std::string affine_str);

    /// Add vertex v at given point p
    void add_vertex(uint v, const Point& p);

    /// Add vertex v at given coordinate x
    void add_vertex(uint v, double x);

    /// Add vertex v at given coordinate (x, y)
    void add_vertex(uint v, double x, double y);

    /// Add vertex v at given coordinate (x, y, z)
    void add_vertex(uint v, double x, double y, double z);

    /// Add vertex v at given point p
    void add_higher_order_vertex(uint v, const Point& p);

    /// Add vertex v at given coordinate x
    void add_higher_order_vertex(uint v, double x);

    /// Add vertex v at given coordinate (x, y)
    void add_higher_order_vertex(uint v, double x, double y);

    /// Add vertex v at given coordinate (x, y, z)
    void add_higher_order_vertex(uint v, double x, double y, double z);

    /// Add cell with given vertices
    void add_cell(uint c, const std::vector<uint>& v);

    /// Add cell (interval) with given vertices
    void add_cell(uint c, uint v0, uint v1);

    /// Add cell (triangle) with given vertices
    void add_cell(uint c, uint v0, uint v1, uint v2);

    /// Add cell (tetrahedron) with given vertices
    void add_cell(uint c, uint v0, uint v1, uint v2, uint v3);

    /// Add higher order cell data (assume P2 triangle for now)
    void add_higher_order_cell_data(uint c, uint v0, uint v1, uint v2, uint v3, uint v4, uint v5);

    /// Close mesh, finish editing, and order entities locally
    void close(bool order=true);

  private:

    // Friends
    friend class TetrahedronCell;

    // Add vertex, common part
    void add_vertex_common(uint v, uint dim);

    // Add higher order vertex, common part
    void add_higher_order_vertex_common(uint v, uint dim);

    // Add cell, common part
    void add_cell_common(uint v, uint dim);

    // Add higher order cell, common part
    void add_higher_order_cell_common(uint v, uint dim);

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

    // NEW HIGHER ORDER MESH STUFF

    // Number of higher order vertices
    uint num_higher_order_vertices;

    // Number of higher order cells <--- should be the same as num_cells!  good for error checking
    uint num_higher_order_cells;

    // Next available higher order vertex
    uint next_higher_order_vertex;

    // Next available higher order cell
    uint next_higher_order_cell;

    // Temporary storage for local higher order cell data
    std::vector<uint> higher_order_cell_data;

  };

}

#endif
