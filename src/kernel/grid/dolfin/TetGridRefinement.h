// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version

#ifndef __TET_GRID_REFINEMENT_H
#define __TET_GRID_REFINEMENT_H

#include <dolfin/GridRefinement.h>

namespace dolfin {

  class Grid;
  class Cell;
  class Face;

  /// Algorithm for the refinement of a tetrahedral grid, based on
  /// the algorithm described in the paper "Tetrahedral Grid Refinement"
  /// by Jürgen Bey, in Computing 55, pp. 355-378 (1995).
  
  class TetGridRefinement : public GridRefinement {
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

    static void refineNoRefine   (Cell& cell, Grid& grid);
    static void refineRegular    (Cell& cell, Grid& grid);
    static void refineIrregular1 (Cell& cell, Grid& grid);
    static void refineIrregular2 (Cell& cell, Grid& grid);
    static void refineIrregular3 (Cell& cell, Grid& grid);
    static void refineIrregular4 (Cell& cell, Grid& grid);

    static void refineIrregular31(Cell& cell, Grid& grid);
    static void refineIrregular32(Cell& cell, Grid& grid,
				  Array<Node*>& sorted_nodes);
    static void refineIrregular33(Cell& cell, Grid& grid, 
				  Array<Node*>& sorted_nodes, Cell& face_neighbor);

    static bool  markedEdgesOnSameFace (Cell& cell);
    static Cell* findNeighbor          (Cell& cell, Face& face);

    static Cell& createCell(Node& n0, Node& n1, Node& n2, Node& n3,
			    Grid& grid, Cell& cell);

  };

}

#endif
