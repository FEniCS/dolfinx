// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NODE_HH
#define __NODE_HH

#include <iostream>
#include <dolfin/constants.h>
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
	 friend std::ostream& operator << (std::ostream& output, const Node& node);
	 void show();
	 
	 /// Friends
	 friend class Grid;
	 friend class Triangle;
	 friend class Tetrahedron;
	 friend class GridData;
	 friend class InitGrid;
	 friend class NodeIterator::NodeNodeIterator;
	 friend class CellIterator::NodeCellIterator;	 
	 
  private:

	 int setID(int id);
	 
	 Point p;	 
	 int _id;
	 
	 // Connectivity
	 ShortList<Node *> nn;
	 ShortList<Cell *> nc;
	 
  };
  
}

#endif
