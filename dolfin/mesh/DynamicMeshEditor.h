// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-22
// Last changed: 2008-09-22

#ifndef __DYNAMIC_MESH_EDITOR_H
#define __DYNAMIC_MESH_EDITOR_H

#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
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
    void addVertex(uint v, const Point& p);

    /// Add vertex v at given coordinate x
    void addVertex(uint v, double x);

    /// Add vertex v at given coordinate (x, y)
    void addVertex(uint v, double x, double y);

    /// Add vertex v at given coordinate (x, y, z)
    void addVertex(uint v, double x, double y, double z);

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
    Array<double> vertex_coordinates;

    // Dynamic storage for cells
    Array<uint> cell_vertices;

  };

}

#endif
