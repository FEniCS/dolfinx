// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_CELL_H
#define __GENERIC_CELL_H

#include <dolfin/constants.h>
#include <dolfin/Cell.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/ShortList.h>

namespace dolfin {

  class Node;
  class Cell;
  
  class GenericCell {
  public:
	 
	 virtual int noNodes() = 0;
	 virtual int noEdges() = 0;
	 virtual int noFaces() = 0;
	 virtual int noBoundaries() = 0;

	 virtual Cell::Type type() = 0;
	 
	 friend class Cell;
	 
  private:

	 virtual bool neighbor(ShortList<Node *> &cn, Cell &cell) = 0;
	 
  };

}

#endif
