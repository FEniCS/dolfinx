// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005-12-01

#ifndef __EDGE_H
#define __EDGE_H

#include <dolfin/dolfin_log.h>
#include <dolfin/PArray.h>
#include <set>
#include <dolfin/VertexIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/FaceIterator.h>

namespace dolfin
{

  class Point;
  class Mesh;
  class EdgeRefData;

  class Edge
  {
  public:
    
    /// Create empty edge
    Edge();
    
    /// Create edge between two given vertices
    Edge(Vertex& n0, Vertex& n1);

    /// Destructor
    ~Edge();

    /// Clear edge data
    void clear();
    
    ///--- Edge data ---
    
    /// Return id of edge
    int  id() const;

    /// Return number of cell neighbors
    unsigned int noCellNeighbors() const;

    /// Get end vertex number i
    Vertex& vertex(int i) const;

    /// Return cell neighbor number i
    Cell& cell(int i) const;

    /// Return the mesh containing the edge
    Mesh& mesh();
    
    /// Return the mesh containing the edge (const version)
    const Mesh& mesh() const;

    /// Get coordinates of vertex number i
    Point& coord(int i) const;
    
    /// Compute and return length of the edge
    real length() const;

    /// Compute and return midpoint of the edge 
    Point midpoint() const;
    
    /// Check if edge consists of the two vertices
    bool equals(const Vertex& n0, const Vertex& n1) const;

    /// Check if edge contains the vertex
    bool contains(const Vertex& n) const;

    /// Check if edge contains the point (point one the same line)
    bool contains(const Point& point) const;

    ///--- Output ---
   
    /// Display condensed edge data
    friend LogStream& operator<<(LogStream& stream, const Edge& edge);
    
    // Friends
    friend class Mesh;
    friend class Vertex;
    friend class Face;
    friend class GenericCell;
    friend class MeshData;
    friend class MeshInit;
    friend class MeshRefinement;
    friend class TriMeshRefinement;
    friend class TetMeshRefinement;
    friend class VertexIterator::CellVertexIterator;
    friend class CellIterator::CellCellIterator;
    friend class EdgeIterator::CellEdgeIterator;
    friend class FaceIterator::CellFaceIterator;
    friend class CellIterator::EdgeCellIterator;
    friend class Triangle;
    friend class Tetrahedron;

    // Boundary information
    std::set<int> ebids;

  private:

    // Specify global edge number
    int setID(int id, Mesh& mesh);
    
    // Set the mesh pointer
    void setMesh(Mesh& mesh);

    /// Specify vertices
    void set(Vertex& n0, Vertex& n1);

    // Initialize marker (if not already done)
    void initMarker();

    // Mark by given cell
    void mark(Cell& cell);

    // Check if cell has been marked for refinement
    bool marked();

    // Check if cell has been marked for refinement by given cell
    bool marked(Cell& cell);

    // Clear marks
    void clearMarks();

    //--- Edge data ---
    
    // The mesh containing this edge
    Mesh* _mesh;
    
    // Global edge number
    int _id;
    
    // Vertices
    Vertex* n0;
    Vertex* n1;

    // Connectivity
    PArray<Cell*> ec;
    
    // Mesh refinement data
    EdgeRefData* rd;

  };
  
}

#endif
