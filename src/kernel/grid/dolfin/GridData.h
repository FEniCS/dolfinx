// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_DATA_H
#define __GRID_DATA_H

#include <dolfin/List.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>

namespace dolfin {

  /// GridData is a container for grid data.
  ///
  /// A block-linked list is used to store the grid data,
  /// constisting of
  ///
  ///    a list of all nodes
  ///    a list of all cells
  ///    a list of all edges

  class GridData {
  public:
	 
	 Node* createNode(int level);
	 Node* createNode(int level, real x, real y, real z);
	 
	 Cell* createCell(int level, Cell::Type type);
	 Cell* createCell(int level, Cell::Type type, int n0, int n1, int n2);
	 Cell* createCell(int level, Cell::Type type, int n0, int n1, int n2, int n3);
	 Cell* createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2);
	 Cell* createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2, Node* n3);

	 void createEdges(Cell* c);

	 void setFinestGridLevel(int gl);
	 int finestGridLevel();

	 Node* getNode(int id);
	 Cell* getCell(int id);
	 Edge* getEdge(int id);

	 int noNodes() const;
	 int noCells() const;
	 int noEdges() const;
	 
	 // Friends
	 friend class NodeIterator::GridNodeIterator;
	 friend class CellIterator::GridCellIterator;
	 friend class EdgeIterator::GridEdgeIterator;
	 
  private:
	 
	 /// Finest grid refinement level
	 int _finest_grid_level;
      
	 List<Node> nodes;
	 List<Cell> cells;
	 List<Edge> edges;
	 
  };
  
}
  
#endif
