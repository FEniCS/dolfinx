// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A couple of comments:
//
//   - More methods should be private?
//   - Why _en0, _en1 instead of just n0, n1?

#ifndef __EDGE_HH
#define __EDGE_HH

#include <dolfin/dolfin_log.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/FaceIterator.h>

namespace dolfin {

  class Point;
  class Grid;

  class Edge{
  public:
    
    /// Create empty edge
    Edge();
    
    /// Create edge between two given nodes
    Edge(Node* n0, Node* n1);

    /// Destructor
    ~Edge();
    
    ///--- Edge data ---
    
    /// Return id of edge
    int  id() const;

    /// Get end node number i
    Node* node(int i) const;

    /// Get coordinates of node number i
    Point coord(int i) const;
    
    /// Compute and return length of the edge
    real length() const;

    /// Compute and return midpoint of the edge 
    Point midpoint() const;
    
    /// Check if edge consists of the two nodes
    bool equals(Node* n0, Node* n1) const;

    ///--- Output ---
   
    /// Display condensed edge data
    friend LogStream& operator<<(LogStream& stream, const Edge& edge);
    
    // Friends
    friend class Grid;
    friend class Node;
    friend class GridData;
    friend class InitGrid;
    friend class NodeIterator::CellNodeIterator;
    friend class CellIterator::CellCellIterator;
    friend class EdgeIterator::CellEdgeIterator;
    friend class FaceIterator::CellFaceIterator;
    friend class Triangle;
    friend class Tetrahedron;
    
  private:
      
    // Specify global edge number
    int setID(int id, Grid* grid);
    
    /// Specify nodes
    void set(Node* n0, Node* n1);
    
    // The grid containing this edge
    Grid* grid;
    
    // Global edge number
    int _id;
    
    // Nodes
    Node* n0;
    Node* n1;
    
    /// --- Functions for mesh refinement
    int  level() const;
    void setLevel(int level);
    void mark();
    void unmark();
    bool marked();
    int  refinedByCells();
    Cell* refinedByCell(int i);
    void  setRefinedByCell(Cell* c);
    void setMarkedForReUse(bool re_use);
    bool markedForReUse();
    
    // FIXME: Remove?
    bool _marked_for_re_use;
    bool marked_for_refinement;
    Array<Cell*> refined_by_cell; 
    int _no_cells_refined;
    // Refinement level in grid hierarchy, coarsest grid is level = 0
    int _level;
    
  };
  
}

#endif
