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

    // Propagate markers for leaf elements
    static void propagateLeafMarks(GridHierarchy& grids);
    
    /// Update marks for edges
    static void updateEdgeMarks(GridHierarchy& grids);

    // The global algorithm
    static void globalRefinement(GridHierarchy& grids);

    // Check consistency of markers before refinement
    static void checkPreCondition(GridHierarchy& grids);

    // Check consistency of markers after refinement
    static void checkPostCondition(GridHierarchy& grids);

    // Check object numbering
    static void checkNumbering(GridHierarchy& grids);

    //--- Algorithms working on the grid on a given level ---
    
    /// Update marks for edges
    static void updateEdgeMarks(Grid& grid);

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
    static void closeCell(Cell& cell, List<Cell*>& cells, Array<bool>& closed);

    /// Check refinement rule for given cell
    static bool checkRule(Cell& cell, int no_marked_edges);
    
    // Refine cell according to refinement rule
    static void refine(Cell& cell, Grid& grid);
    
    ///--- A couple of special functions, placed here rather than in Cell ---

    /// Update marks for edges
    static void updateEdgeMarks(Cell& cell);
    
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

    /// Check if a given cell is a leaf element
    static bool leaf(Cell& cell);
    
    /// Check if cell is allowed to be refined
    static bool okToRefine(Cell& cell);

    /// Create a new node (if it doesn't exist) and set parent-child info
    static Node& createNode(Node& node, Grid& grid, const Cell& cell);
    
    /// Create a new node (if it doesn't exist)
    static Node& createNode(const Point& p, Grid& grid, const Cell& cell);
    
    /// Remove node 
    static void removeNode(Node& node, Grid& grid);

    /// Remove cell 
    static void removeCell(Cell& cell, Grid& grid);

    /// Create a new child to cell, that is a copy of cell 
    static Cell& createChildCopy(Cell& cell, Grid& grid);

  };

}

#endif
