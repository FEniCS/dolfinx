// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_H
#define __GRID_H

// FIXME: remove
#include <stdio.h>

#include <iostream>
#include <dolfin/Variable.h>
#include <dolfin/constants.h>
#include <dolfin/List.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/InitGrid.h>
#include <dolfin/RefineGrid.h>

namespace dolfin {

  class GridData;
  
  class Grid : public Variable {
  public:

	 enum Type { TRIANGLES, TETRAHEDRONS };
	 
	 Grid();
	 Grid(const char *filename);
	 ~Grid();

	 void clear();
	 void refine();
	 
	 int  noNodes() const;
	 int  noCells() const;
	 Type type() const;

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
	 
	 /// Grid data
	 GridData *gd;

	 /// Grid type
	 Type _type;
	 
	 /// Algorithms
	 InitGrid initGrid;
	 RefineGrid refineGrid;
	 
  };

}

#endif
