// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_ITERATOR_H
#define __GRID_ITERATOR_H

#include <dolfin/Array.h>
#include <dolfin/General.h>

namespace dolfin {
  
  class Grid;
  class GridHierarchy;  
  
  typedef Grid* GridPointer;
  
  /// Iterator for the grids in a grid hierarchy.

  class GridIterator {
  public:
    
    /// Create an iterator positioned at the top (coarsest) grid
    GridIterator(const GridHierarchy& grids);

    /// Create an iterator positioned at the given position
    GridIterator(const GridHierarchy& grids, Index index);

    /// Destructor
    ~GridIterator();
   
    /// Step to next grid
    GridIterator& operator++();

    /// Step to previous grid
    GridIterator& operator--();

    /// Check if iterator has reached the first (or last) grid
    bool end();

    /// Return index for current position
    int index();
	 
    operator GridPointer() const;
    Grid& operator*() const;
    Grid* operator->() const;

  private:
    
    Array<Grid*>::Iterator it;
	
  };

}

#endif
