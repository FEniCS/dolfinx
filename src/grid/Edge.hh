// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EDGE_HH
#define __EDGE_HH

// An edge consists of two endpoints.  cell is the geometric part of an element. An element contains
// basis functions, but a cell contains only the geometric information.
//
// Similarly to a Node, the Edge should be small and simple to keep
// the total data size as small as possible.

#include <kw_constants.h>

class Node;
class Point;
class Grid;

class Edge{
public:

  Edge();
  ~Edge();

  void Set(Node *en1, Node *en2, Node *mn);
  void SetEndnodes(Node *en1, Node *en2);
  void SetMidnode(Node *mn);

  /// --- Accessor functions for stored data
  
  /// Get end node number i
  Node* GetEndnode(int i);
  /// Get mid node 
  Node* GetMidnode();

  /// --- Functions that require computation (every time!)
  
  /// Compute and return the lenght of the edge
  real ComputeLength(Grid *grid);
  /// Compute and return midpoint of the edge 
  Point* ComputeMidpoint(Grid *grid);
    
  /// Give access to the special functions below
  friend class Grid;
  friend class Node;
  
protected:
  
private:
  
  Node *end_nodes[2];
  Node *mid_node;
  
};

#endif
