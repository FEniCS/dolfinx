// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TETRAHEDRON_H
#define __TETRAHEDRON_H

#include <dolfin/GenericCell.h>

namespace dolfin {

  class Cell;
  
  class Tetrahedron : public GenericCell {
  public:
	 
	 int noNodes();
	 int noEdges();
	 int noFaces();
	 int noBoundaries();
	 
	 Cell::Type type();
	 
  private:
	 
	 bool neighbor(ShortList<Node *> &cn, Cell &cell);
	 
  };

}

#endif
