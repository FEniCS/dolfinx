// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EDGE_H
#define __EDGE_H

#include <dolfin/dolfin_log.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/FaceIterator.h>

namespace dolfin {

  class Point;
  class Grid;
  class EdgeRefData;

  class Edge{
  public:
    
    /// Create empty edge
    Edge();
    
    /// Create edge between two given nodes
    Edge(Node& n0, Node& n1);

    /// Destructor
    ~Edge();
    
    ///--- Edge data ---
    
    /// Return id of edge
    int  id() const;

    /// Get end node number i
    Node& node(int i) const;

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

    ///--- Output ---
   
    /// Display condensed edge data
    friend LogStream& operator<<(LogStream& stream, const Edge& edge);
    
    // Friends
    friend class Grid;
    friend class Node;
    friend class GridData;
    friend class GridInit;
    friend class GridRefinement;
    friend class TriGridRefinement;
    friend class TetGridRefinement;
    friend class NodeIterator::CellNodeIterator;
    friend class CellIterator::CellCellIterator;
    friend class EdgeIterator::CellEdgeIterator;
    friend class FaceIterator::CellFaceIterator;
    friend class Triangle;
    friend class Tetrahedron;
    
  private:
      
    // Specify global edge number
    int setID(int id, Grid& grid);
    
    // Set the grid pointer
    void setGrid(Grid& grid);

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

    //--- Edge data ---
    
    // The grid containing this edge
    Grid* grid;
    
    // Global edge number
    int _id;
    
    // Nodes
    Node* n0;
    Node* n1;

    // Grid refinement data
    EdgeRefData* rd;

  };
  
}

#endif
