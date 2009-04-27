// Copyright (C) 2006-20008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-16
// Last changed: 2008-11-13

#ifndef __MESH_EDITOR_H
#define __MESH_EDITOR_H

#include <vector>
#include <dolfin/common/types.h>
#include "CellType.h"

namespace dolfin
{

  class Mesh;
  class Point;
  class Vector;

  /// A simple mesh editor for creating simplicial meshes in 1D, 2D and 3D.

  class MeshEditor
  {
  public:

    /// Constructor
    MeshEditor();

    /// Destructor
    ~MeshEditor();

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, CellType::Type type, uint tdim, uint gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, std::string type, uint tdim, uint gdim);

    /// Specify number of vertices
    void init_vertices(uint num_vertices);

    /// Specify number of vertices
    void initHigherOrderVertices(uint num_higher_order_vertices);

    /// Specify number of cells
    void init_cells(uint num_cells);

    /// Specify number of cells
    void initHigherOrderCells(uint num_higher_order_cells, uint num_higher_order_cell_dof);

    /// Set boolean indicator inside MeshGeometry
    void setAffineCellIndicator(uint c, const std::string affine_str);

    /// Add vertex v at given point p
    void add_vertex(uint v, const Point& p);

    /// Add vertex v at given coordinate x
    void add_vertex(uint v, double x);

    /// Add vertex v at given coordinate (x, y)
    void add_vertex(uint v, double x, double y);

    /// Add vertex v at given coordinate (x, y, z)
    void add_vertex(uint v, double x, double y, double z);

    /// Add vertex v at given point p
    void addHigherOrderVertex(uint v, const Point& p);

    /// Add vertex v at given coordinate x
    void addHigherOrderVertex(uint v, double x);

    /// Add vertex v at given coordinate (x, y)
    void addHigherOrderVertex(uint v, double x, double y);

    /// Add vertex v at given coordinate (x, y, z)
    void addHigherOrderVertex(uint v, double x, double y, double z);

    /// Add cell with given vertices
    void add_cell(uint c, const std::vector<uint>& v);

    /// Add cell (interval) with given vertices
    void add_cell(uint c, uint v0, uint v1);

    /// Add cell (triangle) with given vertices
    void add_cell(uint c, uint v0, uint v1, uint v2);

    /// Add cell (tetrahedron) with given vertices
    void add_cell(uint c, uint v0, uint v1, uint v2, uint v3);

    /// Add higher order cell data (assume P2 triangle for now)
    void addHigherOrderCellData(uint c, uint v0, uint v1, uint v2, uint v3, uint v4, uint v5);

    /// Close mesh, finish editing, and order entities locally
    void close(bool order=true);

  private:

    // Add vertex, common part
    void add_vertexCommon(uint v, uint dim);

    // Add higher order vertex, common part
    void addHigherOrderVertexCommon(uint v, uint dim);

    // Add cell, common part
    void add_cellCommon(uint v, uint dim);

    // Add higher order cell, common part
    void addHigherOrderCellCommon(uint v, uint dim);

    // Clear all data
    void clear();

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
