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
	 
	 virtual int noNodes() const = 0;
	 virtual int noEdges() const = 0;
	 virtual int noFaces() const = 0;
	 virtual int noBound() const = 0;

	 virtual Cell::Type type() const = 0;

	 friend class Cell;
	 
  private:
	 
	 virtual bool neighbor(ShortList<Node *> &cn, Cell &cell) const = 0;
	 
  };

}

#endif
