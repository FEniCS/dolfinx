// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_ITERATORS_H
#define __GRID_ITERATORS_H

#include <dolfin/List.h>

namespace dolfin {

  class Grid;
  class Node;
  class Cell;
  class GenericNodeIterator;
  class GenericCellIterator;
  
  // General iterator for nodes
  class NodeIterator {
  public:
	 
	 NodeIterator(Grid& grid);
	 ~NodeIterator();
	 
	 void operator++();
	 bool end();
	 
	 Node& operator*() const;
	 Node* operator->() const;
	 
  private:
	 
	 GenericNodeIterator *n;
	 
  };

  // General iterator for cells
  class CellIterator {
  public:
	 
	 CellIterator(Grid& grid);
	 ~CellIterator();

	 void operator++();
	 bool end();
	 
	 Cell& operator*() const;
	 Cell* operator->() const;
	 
  private:
	 
	 GenericCellIterator *c;

  };

  // Base class for node iterators
  class GenericNodeIterator {
  public:

	 virtual void operator++() = 0;
	 virtual bool end() = 0;

	 virtual Node& operator*() const = 0;
	 virtual Node* operator->() const = 0;
	 virtual Node* pointer() const = 0;
	 
  };

  // Base class for cell iterators
  class GenericCellIterator {
  public:
	 
	 virtual void operator++() = 0;
	 virtual bool end() = 0;
	 
	 virtual Cell& operator*() const = 0;
	 virtual Cell* operator->() const = 0;
	 virtual Cell* pointer() const = 0;
	 
  };
  
  // Iterator for the nodes in a grid
  class GridNodeIterator : public GenericNodeIterator {
  public:
	 
	 GridNodeIterator(Grid& grid); 
	 
	 void operator++();
	 bool end();
	 
	 Node& operator*() const;
	 Node* operator->() const;
	 Node* pointer() const;
	 
  private:

	 List<Node>::Iterator node_iterator;
	 List<Node>::Iterator at_end;
	 
  };

  // Iterator for the cells in a grid
  class GridCellIterator : public GenericCellIterator {
  public:

	 GridCellIterator(Grid& grid); 

 	 void operator++();
	 bool end();
	 
	 Cell& operator*() const;
	 Cell* operator->() const;
	 Cell* pointer() const;
	 
  private:

	 List<Cell>::Iterator cell_iterator;
	 List<Cell>::Iterator at_end;
	 
  };
  
}

#endif
