// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_REFINEMENT_DATA_H
#define __GRID_REFINEMENT_DATA_H

#include <dolfin/CellMarker.h>
#include <dolfin/Array.h>
#include <dolfin/List.h>

namespace dolfin {

  class Cell;
  class Grid;

  /// GridRefinementData is a container for data created during
  /// grid refinement. This data is only created during refinement
  /// and does not need to be stored for grids that are not refined.

  class GridRefinementData {
  public:
    
    /// Create an empty set of grid refinement data
    GridRefinementData(Grid* grid);

    /// Destructor
    ~GridRefinementData();

    /// Clear all data
    void clear();

    /// Mark cell for refinement
    void mark(Cell* cell);
    
    /// Return number of cells marked for refinement
    int noMarkedCells() const;

  private:

    // The grid
    Grid* grid;

    // Cells marked for refinement
    List<Cell*> marked_cells;

  };

}

#endif
