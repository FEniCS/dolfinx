// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version

#ifndef __GRID_REFINEMENT_H
#define __GRID_REFINEMENT_H

#include <dolfin/List.h>

namespace dolfin {

  class Grid;
  class Cell;
  class GridHierarchy;

  /// Algorithm for the refinement of a triangular or tetrahedral grid.
  
  class GridRefinement {
  public:

    /// Refine a given grid hierarchy according to marks
    static void refine(GridHierarchy& grids);
    
  protected:
    
    //--- Algorithms working on the whole grid hierarchy ---

    // The global algorithm
    static void globalRefinement(GridHierarchy& grids);

    //--- Algorithms working on the grid on a given level ---

    // Set initial marks for cells and edges
    static void initMarks(Grid& grid);

    // Evaluate and adjust marks for a grid
    static void evaluateMarks(Grid& grid);

    // Perform the green closer on a grid
    static void closeGrid(Grid& grid);

    // Refine a grid according to marks
    static void refineGrid(Grid& grid);

    // Unrefine a grid according to marks
    static void unrefineGrid(Grid& grid, const GridHierarchy& grids);

    ///--- Algorithms working on a given cell ---
    
    // Close a cell
    static void closeCell(Cell& cell, List<Cell*>& cells);

    /// Check refinement rule for given cell
    static bool checkRule(Cell& cell, int no_marked_edges);
    
    // Refine cell according to refinement rule
    static void refine(Cell& cell, Grid& grid);
    
    ///--- A couple of special functions, placed here rather than in Cell ---

    /// Check if all children are marked for coarsening
    static bool childrenMarkedForCoarsening(Cell& cell);

    /// Check if at least one edge of a child is marked for refinement
    static bool edgeOfChildMarkedForRefinement(Cell& cell);

    /// Check if the cell has at least one edge marked by another cell (but not the cell itself)
    static bool edgeMarkedByOther(Cell& cell);

    /// Sort nodes, placing the node belonging to the most number of marked edges first
    static void sortNodes(const Cell& cell, Array<Node*>& nodes);
    
    /// Count the number of marked edges within a cell
    static int noMarkedEdges(const Cell& cell);

    /// Mapping from global node number to local number within cell
    static int nodeNumber(const Node& node, const Cell& cell);

  };

}

#endif
