// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include <dolfin/GenericCell.h>

namespace dolfin {

  class Cell;
  
  class Triangle : public GenericCell {
  public:
	 
	 int noNodes() const;
	 int noEdges() const;
	 int noFaces() const;
	 int noBound() const;
	 
	 Cell::Type type() const;

  private:

	 bool neighbor(ShortList<Node *> &cn, Cell &cell) const;

  };

}

#endif
