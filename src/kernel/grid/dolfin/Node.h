// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NODE_HH
#define __NODE_HH

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/Point.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>

namespace dolfin{

  class GenericCell;
  class Cell;
  class Edge;
  class GridInit;
  
  class Node{
  public:

    /// Create an unconnected node at (0,0,0)
    Node();

    /// Create an unconnected node at (x,0,0)
    Node(real x);
    
    /// Create an unconnected node at (x,y,0)
    Node(real x, real y);

    /// Create an unconnected node at (x,y,z)
    Node(real x, real y, real z);
    
    ///--- Node data ---

    /// Return id of node
    int id() const;

    /// Return number of node neighbors
    int noNodeNeighbors() const;

    /// Return number of cell neighbors
    int noCellNeighbors() const;

    /// Return number of edge neighbors
    int noEdgeNeighbors() const;

    /// Return node neighbor number i
    Node* node(int i) const;

    /// Return cell neighbor number i
    Cell* cell(int i) const;

    /// Return edge neighbor number i
    Edge* edge(int i) const;

    /// Return node coordinate
    Point coord() const;
    
    /// Return distance to given node
    real dist(const Node& n) const;    
    
    /// Check if given node is a neighbor
    bool neighbor(Node* n);

    /// Return boundary number for node
    int boundary() const;

    /// Comparison based on the node id

    bool operator== (int id) const;
    bool operator<  (int id) const;
    bool operator<= (int id) const;
    bool operator>  (int id) const;
    bool operator>= (int id) const;

    friend bool operator== (int id, const Node& node);
    friend bool operator<  (int id, const Node& node);
    friend bool operator<= (int id, const Node& node);
    friend bool operator>  (int id, const Node& node);
    friend bool operator>= (int id, const Node& node);

    ///--- Output ---
   
    /// Display condensed node data
    friend LogStream& operator<<(LogStream& stream, const Node& node);
    
    /// Friends
    friend class Grid;
    friend class Triangle;
    friend class Tetrahedron;
    friend class GridData;
    friend class GridInit;
    friend class NodeIterator::NodeNodeIterator;
    friend class CellIterator::NodeCellIterator;	 
    friend class EdgeIterator::NodeEdgeIterator;	 
    
  private:

    // Specify global node number
    int setID(int id, Grid* grid);
    
    // Specify coordinate
    void set(real x, real y, real z);
    
    // The grid containing this node
    Grid* grid;

    // Global node number
    int _id;

    // Node coordinate
    Point p;

    // Connectivity
    Array<Node*> nn;
    Array<Cell*> nc;
    Array<Edge*> ne;





    // FIXME: Remove?
    int _boundary;
    Node* _child;    

    int level() const;
    Node* child();
    void setMarkedForReUse(bool re_use);
    bool markedForReUse();
    void setChild(Node* child);

    // FIXME: Remove?
    // Refinement level in grid hierarchy, coarsest grid is level = 0
    int _level;
    void setLevel(int level);
    bool _marked_for_re_use;

  };
  
}

#endif
