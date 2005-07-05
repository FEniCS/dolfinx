// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005

#ifndef __EDGE_H
#define __EDGE_H

#include <dolfin/dolfin_log.h>
#include <dolfin/PArray.h>
#include <set>
#include <dolfin/NodeIterator.h>
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
    
    /// Create edge between two given nodes
    Edge(Node& n0, Node& n1);

    /// Destructor
    ~Edge();

    /// Clear edge data
    void clear();
    
    ///--- Edge data ---
    
    /// Return id of edge
    int  id() const;

    /// Return number of cell neighbors
    unsigned int noCellNeighbors() const;

    /// Get end node number i
    Node& node(int i) const;

    /// Return cell neighbor number i
    Cell& cell(int i) const;

    /// Return the mesh containing the edge
    Mesh& mesh();
    
    /// Return the mesh containing the edge (const version)
    const Mesh& mesh() const;

    /// Get coordinates of node number i
    Point& coord(int i) const;
    
    /// Compute and return length of the edge
    real length() const;

    /// Compute and return midpoint of the edge 
    Point midpoint() const;
    
    /// Check if edge consists of the two nodes
    bool equals(const Node& n0, const Node& n1) const;

    /// Check if edge contains the node
    bool contains(const Node& n) const;

    /// Check if edge contains the point (point one the same line)
    bool contains(const Point& point) const;

    ///--- Output ---
   
    /// Display condensed edge data
    friend LogStream& operator<<(LogStream& stream, const Edge& edge);
    
    // Friends
    friend class Mesh;
    friend class Node;
    friend class Face;
    friend class GenericCell;
    friend class MeshData;
    friend class MeshInit;
    friend class MeshRefinement;
    friend class TriMeshRefinement;
    friend class TetMeshRefinement;
    friend class NodeIterator::CellNodeIterator;
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

    /// Specify nodes
    void set(Node& n0, Node& n1);

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
    
    // Nodes
    Node* n0;
    Node* n1;

    // Connectivity
    PArray<Cell*> ec;
    
    // Mesh refinement data
    EdgeRefData* rd;

  };
  
}

#endif
