// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-16
// Last changed: 2006-06-05

#ifndef __MESH_EDITOR_H
#define __MESH_EDITOR_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>

namespace dolfin
{
  
  class NewMesh;
  
  /// A simple mesh editor for creating simplicial meshes in 1D, 2D and 3D.

  class MeshEditor
  {
  public:
    
    /// Constructor
    MeshEditor();
    
    /// Destructor
    ~MeshEditor();

    /// Edit given mesh
    void edit(NewMesh& mesh, uint dim, std::string cell_type);

    /// Specify number of vertices
    void initVertices(uint num_vertices);

    /// Specify number of cells
    void initCells(uint num_cells);

    /// Add vertex v at given coordinate x
    void addVertex(uint v, real x);

    /// Add vertex v at given coordinate (x, y)
    void addVertex(uint v, real x, real y);

    /// Add vertex v at given coordinate (x, y, z)
    void addVertex(uint v, real x, real y, real z);

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

    // Topological and Euclidean dimension (assumed equal here)
    uint dim;
    
    // Number of vertices
    uint num_vertices;

    // Number of cells
    uint num_cells;

    // Next available vertex
    uint next_vertex;

    // Next available cell
    uint next_cell;

    // The mesh
    NewMesh* mesh;

    // Temporary storage for local cell data
    Array<uint> vertices;
   
  };

}

#endif
