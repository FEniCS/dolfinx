// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_ITERATOR_H
#define __GRID_ITERATOR_H

#include <dolfin/Array.h>

namespace dolfin {
  
  class Grid;
  class GridHierarchy;  
  
  typedef Grid* GridPointer;
  
  /// Iterator for the grids in a grid hierarchy.

  class GridIterator {
  public:
    
    GridIterator(const GridHierarchy& grids);
    ~GridIterator();
	
    GridIterator& operator++();
    GridIterator& operator--();

    bool end();
    int index();
	 
    operator GridPointer() const;
    Grid& operator*() const;
    Grid* operator->() const;

  private:
    
    Array<Grid*>::Iterator it;
	
  };

}

#endif
