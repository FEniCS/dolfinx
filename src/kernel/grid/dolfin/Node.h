// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NODE_HH
#define __NODE_HH

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Point.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/ShortList.h>

namespace dolfin{

  class GenericCell;
  class Cell;
  class Edge;
  class InitGrid;
  
  class Node{
  public:
    
    Node();
    Node(real x);
    Node(real x, real y);
    Node(real x, real y, real z);
    
    void  set(real x, real y, real z);
    int   id() const;
    Point coord() const;
    int level() const;
    Node* child();
    Edge* edge(int i);
    Cell* cell(int i); 
    int   boundary() const;
    bool  neighbor(Node* n);
    
    void setChild(Node* child);

    int   noNodeNeighbors() const;
    int   noCellNeighbors() const;
    int   noEdgeNeighbors() const;
 
    // Comparison with numbers based on id

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
   
    /// Output
    friend LogStream& operator<<(LogStream& stream, const Node& node);
    void show();
    
    /// Friends
    friend class Grid;
    friend class Triangle;
    friend class Tetrahedron;
    friend class GridData;
    friend class InitGrid;
    friend class NodeIterator::NodeNodeIterator;
    friend class CellIterator::NodeCellIterator;	 
    friend class EdgeIterator::NodeEdgeIterator;	 
    
  private:
    
    int setID(int id);
    
    // Node data
    Point p;	 
    int _id;
    int _boundary;
    
    // Refinement level in grid hierarchy, coarsest grid is level = 0
    int _level;
    void setLevel(int level);

    // Connectivity
    ShortList<Node*> nn;
    ShortList<Cell*> nc;
    ShortList<Edge*> ne;
    Node* _child;

  };
  
}

#endif
