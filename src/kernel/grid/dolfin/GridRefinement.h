// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A couple of comments:
//
//   - Return List<Cell *> from closeCell() won't work?
//   - Rename refineGrid(int level) and unrefineGrid(int level) to refine(int level) and unrefine(int level)?
//   - Should there be an unrefine()?

#ifndef __GRID_REFINEMENT_H
#define __GRID_REFINEMENT_H

namespace dolfin {

  class Grid;
  class Edge;
  class Cell;
  class GridHierarchy;

  /// Algorithm for the refinement of a triangular or tetrahedral grid.
  ///
  /// Based on the algorithm described in the paper "Tetrahedral Grid Refinement"
  /// by Jürgen Bey, in Computing 55, pp. 355-378 (1995).
  
  class GridRefinement {
  public:

    /// Refine a given grid hierarchy according to marks
    static void refine(GridHierarchy& grids);
    
  private:
    
    //--- Algorithms working on the whole grid hierarchy ---

    // Create the new finest grid
    static void createFineGrid(GridHierarchy& grids);

    // The global algorithm working on the whole grid hierarchy
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
    static void unrefineGrid(Grid& grid);

    ///--- Algorithms working on a given cell ---
    
    // Close a cell
    static void closeCell(Cell& cell);

    // Refine a cell regularly
    static void regularRefinement(Cell& cell, Grid& grid);
    
    // Refine a triangle regularly
    static void regularRefinementTri(Cell& cell, Grid& grid);

    // Refine a tetrahedron regularly
    static void regularRefinementTet(Cell& cell, Grid& grid);

    // Refine a tetrahedron irregularly
    // (different cases depending on number of marked edges)
    void irregTetRefByRule1(Cell& cell, Grid& grid);
    void irregTetRefByRule2(Cell& cell, Grid& grid);
    void irregTetRefByRule3(Cell& cell, Grid& grid);
    void irregTetRefByRule4(Cell& cell, Grid& grid);

    ///--- A couple of special functions, placed here rather than in Cell ---

    /// Check if all children are marked for coarsening
    static bool childrenMarkedForCoarsening(Cell& cell);

    /// Check if at least one edge of a child is marked for refinement
    static bool oneEdgeOfChildMarkedForRefinement(Cell& cell);

    /// Check if the cell has at least on edge that is marked
    static bool oneEdgeMarkedForRefinement(Cell& cell);
    

 

    /*
    void localIrregularRefinement(Cell *parent);
    */
    
    

  };

}

#endif
