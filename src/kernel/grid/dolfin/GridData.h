// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_DATA_H
#define __GRID_DATA_H

#include <dolfin/List.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>

namespace dolfin {

  /// GridData is a container for grid data.
  ///
  /// A block-linked list is used to store the grid data,
  /// constisting of
  ///
  ///    a list of all nodes
  ///    a list of all cells

  class GridData {
  public:
	 
	 Node* createNode();
	 Node* createNode(real x, real y, real z);
	 
	 Cell* createCell(int level, Cell::Type type);
	 Cell* createCell(int level, Cell::Type type, int n0, int n1, int n2);
	 Cell* createCell(int level, Cell::Type type, int n0, int n1, int n2, int n3);
	 Cell* createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2);
	 Cell* createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2, Node* n3);

	 Node* getNode(int id);
	 Cell* getCell(int id);

	 int noNodes() const;
	 int noCells() const;
	 
	 // Friends
	 friend class NodeIterator::GridNodeIterator;
	 friend class CellIterator::GridCellIterator;
	 
  private:
	 
	 List<Node> nodes;
	 List<Cell> cells;
	 
  };
  
}
  
#endif
