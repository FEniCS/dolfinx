// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NODE_HH
#define __NODE_HH

#include <kw_constants.h>

#include "Point.hh"

class Cell;
class Grid;

class Node{
public:

  Node();
  ~Node();

  void Clear();
  
  /// --- Neigbor information

  /// Get number of cell neighbors
  int GetNoCellNeighbors();
  /// Get cell neighbor number i
  int GetCellNeighbor(int i);
  /// Get number of node neighbors
  int GetNoNodeNeighbors();
  /// Get node neighbor number i
  int GetNodeNeighbor(int i);

  /// --- Accessor functions for stored data

  /// Set coordinates
  void SetCoord(float x, float y, float z);  
  /// Get coordinate i
  real GetCoord(int i);
  /// Get all coordinates
  Point GetCoord();

  /// Give access to the special functions below
  friend class Grid;
  friend class Triangle;
  friend class Tetrahedron;

private:
  
  /// Member functions used for computing neighbor information
  
  /// Allocate memory for list of neighbor cells
  void AllocateForNeighborCells();
  /// Check if this and the other node have a common cell neighbor
  bool CommonCell(Node *n, int thiscell, int *cellnumber);
  /// Check if this and the other two nodes have a common cell neighbor
  bool CommonCell(Node *n1, Node *n2, int thiscell, int *cellnumber);
  /// Return an upper bound for the number of node neighbors
  int GetMaxNodeNeighbors(Cell **cell_list);
  /// Compute node neighbors of the node
  void ComputeNodeNeighbors(Cell **cell_list, int thisnode, int *tmp);

  Point p;

  int *neighbor_nodes;
  int *neighbor_cells;
  int nn;
  int nc;
  
};

#endif
