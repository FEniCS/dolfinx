// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_HIERARCHY
#define __GRID_HIERARCHY

#include <dolfin/Array.h>

namespace dolfin {

  class Grid;

  class GridHierarchy {
  public:

    /// Create empty grid hierarchy
    GridHierarchy();

    /// Create a grid hierarchy from a given grid
    GridHierarchy(Grid& grid);

    /// Destructor
    ~GridHierarchy();

    /// Compute grid hierarchy from a given grid
    void init(Grid& grid);

    /// Clear grid hierarchy
    void clear();

    /// Add a grid to the grid hierarchy
    void add(Grid& grid);

    /// Return grid at given level
    Grid& operator() (int level) const;

    /// Return coarsest grid (level 0)
    Grid& coarse() const;

    /// Return finest grid (highest level)
    Grid& fine() const;

    /// Return number of levels
    int size() const;

    /// Check if grid hierarchy is empty
    bool empty() const;

    /// Friends
    friend class GridIterator;

  private:

    // An array of grid pointers
    Array<Grid*> grids;

  };

}

#endif
