// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_DATA_H
#define __GRID_DATA_H

/// GridData is a container for grid data.
///
/// Block linked list is used to store the grid data,
/// constisting of
///
///    a list of all nodes
///    a list of all cells

#include <dolfin/List.h>
#include <dolfin/Cell.h>

namespace dolfin {

  class Node;
  class Triangle;
  class Tetrahedron;
  
  class GridData {
  public:
	 
	 GridData();
	 ~GridData();
	 
	 Node* createNode();
	 Cell* createCell(Cell::Type type);

	 Node* createNode(real x, real y, real z);
	 Cell* createCell(Cell::Type type, int n0, int n1, int n2);
	 Cell* createCell(Cell::Type type, int n0, int n1, int n2, int n3);

	 Node* getNode(int id);
	 Cell* getCell(int id);

	 // Friends
	 friend class GridNodeIterator;
	 friend class GridCellIterator;
	 
  private:
	 
	 List<Node> nodes;
	 List<Cell> cells;
	 
  };
  
}
  
#endif
