// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BOUNDARY_H
#define __BOUNDARY_H

#include <dolfin/BoundaryData.h>

namespace dolfin {

  class Grid;

  class Boundary {
  public:
    
    /// Create a boundary for given grid
    Boundary(Grid& grid);

    /// Destructor
    ~Boundary();

    /// Friends
    friend class NodeIterator::BoundaryNodeIterator;

  private:

    // Compute boundary (and clear old data)
    void init();

    // Clear boundary
    void clear();
    
    // The grid
    Grid* grid;

  };

}

#endif
