// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NODE_HH
#define __NODE_HH

#include <iostream>
#include <dolfin/dolfin_constants.h>
#include <dolfin/Point.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/ShortList.h>

namespace dolfin{

  class GenericCell;
  class Cell;
  class InitGrid;
  
  class Node{
  public:
	 
	 Node();
	 ~Node();
	 
	 void  set(real x, real y, real z);
	 int   id() const;
	 Point coord() const;
	 
	 /// Output
	 void show();
	 friend std::ostream& operator << (std::ostream& output, const Node& node);

	 // old functions below
	 
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
	 
	 /// Set global node number
	 void SetNodeNo(int nn);  
	 /// Set coordinates
	 void SetCoord(float x, float y, float z);  
	 /// Get global node number
	 int GetNodeNo();  
	 /// Get coordinate i
	 real GetCoord(int i);
	 /// Get all coordinates
	 Point* GetCoord();
	 
	 /// Friends
	 friend class Grid;
	 friend class Triangle;
	 friend class Tetrahedron;
	 friend class GridData;
	 friend class InitGrid;
	 friend class NodeIterator::NodeNodeIterator;
	 friend class CellIterator::NodeCellIterator;
	 
	 
  protected:
	
	 /// Member functions used for computing neighbor information
	 
	 /// Allocate memory for list of neighbor cells
	 void AllocateForNeighborCells();
	 /// Check if this and the other node have a common cell neighbor
	 bool CommonCell(Node *n, int thiscell, int *cellnumber);
	 /// Check if this and the other two nodes have a common cell neighbor
	 bool CommonCell(Node *n1, Node *n2, int thiscell, int *cellnumber);
	 /// Return an upper bound for the number of node neighbors
	 int GetMaxNodeNeighbors(GenericCell **cell_list);
	 /// Compute node neighbors of the node
	 void ComputeNodeNeighbors(GenericCell **cell_list, int thisnode, int *tmp);
	 
  private:

	 int setID(int id);
	 void clear();
	 
	 Point p;
	 
	 int _id;
	 
	 int global_node_number;
	 
	 int *neighbor_nodes;
	 int *neighbor_cells;
	 int _nn;
	 int _nc;

	 // Connectivity
	 ShortList<Node *> nn;
	 ShortList<Cell *> nc;
	 
  };
  
}

#endif
