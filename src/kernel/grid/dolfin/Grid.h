// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_H
#define __GRID_H

// FIXME: remove
#include <stdio.h>

#include <iostream>

#include <dolfin/dolfin_constants.h>
#include <dolfin/List.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>

namespace dolfin {

  class GridData;
  
  class Grid {
  public:
	 
	 Grid();
	 ~Grid();

	 void clear();
	 
	 int noNodes();
	 int noCells();

	 /// Output
	 void show();
	 friend std::ostream& operator << (std::ostream& output, Grid& grid);
	 
	 /// Friends
	 friend class NodeIterator::GridNodeIterator;
	 friend class CellIterator::GridCellIterator;
	 friend class XMLGrid;
	 
  private:
	 
 	 Node* createNode();
	 Cell* createCell(Cell::Type type);

	 Node* createNode(real x, real y, real z);
	 Cell* createCell(Cell::Type type, int n0, int n1, int n2);
	 Cell* createCell(Cell::Type type, int n0, int n1, int n2, int n3);

	 Node* getNode(int id);
	 Cell* getCell(int id);

	 void init();
	 
	 /// --- Grid data (main part) ---

	 GridData *grid_data;
	 int no_nodes;
	 int no_cells;

  };

}

#endif
