// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-10
// Last changed: 2006

#ifndef __BOUNDARY_ITERATOR_H
#define __BOUNDARY_ITERATOR_H

#include <dolfin/PArray.h>
#include <dolfin/PList.h>
#include <dolfin/Table.h>

namespace dolfin {

  template <class T, class V>
  class BoundaryIterator {
  public:
	 
    /// Constructor
    BoundaryIterator(Boundary& boundary) : iterator(boundary){}

    /// Constructor
    BoundaryIterator(Boundary* boundary) : iterator(boundary){}

    /// Destructor
    ~BoundaryIterator(){}

    bool end()
      { return iterator.end(); }

    T& operator++()
      { return ++iterator; }

    V* operator->()
      { return iterator.pointer(); } 

    const uint numCellNeighbors() 
      { return iterator->numCellNeighbors(); } 

    Cell& cell(uint cell) 
      { return iterator->cell(cell); } 

    bool contains(const Point& p) 
      { return iterator->contains(p); }

  private:

    T iterator;
	 
  };

}

#endif
