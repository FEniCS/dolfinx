// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version

#ifndef __TRI_GRID_REFINEMENT_H
#define __TRI_GRID_REFINEMENT_H

#include <dolfin/GridRefinement.h>

namespace dolfin {

  class Grid;
  class Cell;

  /// Algorithm for the refinement of a triangular grid, a modified version
  /// of the algorithm described in the paper "Tetrahedral Grid Refinement"
  /// by Jürgen Bey, in Computing 55, pp. 355-378 (1995).
  
  class TriGridRefinement : public GridRefinement {
  public:

    /// Choose refinement rule
    static bool checkRule(Cell& cell, int no_marked_edges);

    /// Refine according to rule
    static void refine(Cell& cell, Grid& grid);

  private:

    static bool checkRuleRegular   (Cell& cell, int no_marked_edges);
    static bool checkRuleIrregular1(Cell& cell, int no_marked_edges);
    static bool checkRuleIrregular2(Cell& cell, int no_marked_edges);
    static bool checkRuleIrregular3(Cell& cell, int no_marked_edges);
    static bool checkRuleIrregular4(Cell& cell, int no_marked_edges);

    static void refineRegular   (Cell& cell, Grid& grid);
    static void refineIrregular1(Cell& cell, Grid& grid);
    static void refineIrregular2(Cell& cell, Grid& grid);
    static void refineIrregular3(Cell& cell, Grid& grid);
    static void refineIrregular4(Cell& cell, Grid& grid);

  };

}

#endif
