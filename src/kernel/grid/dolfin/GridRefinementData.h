// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_REFINEMENT_DATA_H
#define __GRID_REFINEMENT_DATA_H

#include <dolfin/Array.h>
#include <dolfin/List.h>
#include <dolfin/CellRefData.h>
#include <dolfin/EdgeRefData.h>

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
    
    /// Friends
    friend class Grid;
    friend class GridRefinement;

  private:

    // Change the grid pointer
    void setGrid(Grid& grid);

    // Initialize markers
    void initMarkers();

    // Return cell marker
    Cell::Marker& cellMarker(int id);
    
    // Mark edge by given cell
    void edgeMark(int id, Cell& cell);

    // Check if edge is marked
    bool edgeMarked(int id) const;

    // Check if edge is marked by given cell
    bool edgeMarked(int id, Cell& cell) const;

    //--- Grid refinement data ---
    
    // The grid
    Grid* grid;

    // Cells marked for refinement
    List<Cell*> marked_cells;

    // Cell markers
    Array<CellRefData> cell_markers;
    Array<EdgeRefData> edge_markers;

  };

}

#endif
