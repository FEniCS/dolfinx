// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CELL_H
#define __CELL_H

#include <iostream>

namespace dolfin {  

  class Node;
  class GenericCell;
   
  class Cell {
  public:

	 Cell();
	 ~Cell();

	 enum Type { TRIANGLE, TETRAHEDRON, NONE };
 
	 int id() const;
	 Cell::Type type() const;
	 
	 void set(Node *n0, Node *n1, Node *n2);
	 void set(Node *n0, Node *n1, Node *n2, Node *n3);
 
	 /// Output
	 friend std::ostream& operator << (std::ostream& output, const Cell& cell);
	 
	 // Friends
	 friend class GridData;
	 friend class InitGrid;
	 
  private:

	 void setID(int id);
	 void init(Type type);
	 void clear();
	 
	 GenericCell *c;
	 
  };

}

#endif
