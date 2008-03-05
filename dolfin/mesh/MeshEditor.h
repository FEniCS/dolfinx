// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-16
// Last changed: 2006-11-17

#ifndef __MESH_EDITOR_H
#define __MESH_EDITOR_H

#include <dolfin/main/constants.h>
#include "CellType.h"
#include <dolfin/common/Array.h>

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

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, CellType::Type type, uint tdim, uint gdim);

    /// Open mesh of given cell type, topological and geometrical dimension
    void open(Mesh& mesh, std::string type, uint tdim, uint gdim);

    /// Specify number of vertices
    void initVertices(uint num_vertices);

    /// Specify number of cells
    void initCells(uint num_cells);

    /// Add vertex v at given point p
    void addVertex(uint v, const Point& p);

    /// Add vertex v at given coordinate x
    void addVertex(uint v, real x);

    /// Add vertex v at given coordinate (x, y)
    void addVertex(uint v, real x, real y);

    /// Add vertex v at given coordinate (x, y, z)
    void addVertex(uint v, real x, real y, real z);

    /// Add cell with given vertices
    void addCell(uint c, const Array<uint>& v);

    /// Add cell (interval) with given vertices
    void addCell(uint c, uint v0, uint v1);

    /// Add cell (triangle) with given vertices
    void addCell(uint c, uint v0, uint v1, uint v2);
    
    /// Add cell (tetrahedron) with given vertices
    void addCell(uint c, uint v0, uint v1, uint v2, uint v3);

    /// Close mesh, finish editing
    void close();

  private:

    // Add vertex, common part
    void addVertexCommon(uint v, uint dim);

    // Add cell, common part
    void addCellCommon(uint v, uint dim);

    // Clear all data
    void clear();

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

    // The mesh
    Mesh* mesh;

    // Temporary storage for local cell data
    Array<uint> vertices;
   
  };

}

#endif
