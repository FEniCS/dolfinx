// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version

#ifndef __TRI_MESH_REFINEMENT_H
#define __TRI_MESH_REFINEMENT_H

#include <dolfin/MeshRefinement.h>

namespace dolfin {

  class Mesh;
  class Cell;

  /// Algorithm for the refinement of a triangular mesh, a modified version
  /// of the algorithm described in the paper "Tetrahedral Mesh Refinement"
  /// by Jürgen Bey, in Computing 55, pp. 355-378 (1995).
  
  class TriMeshRefinement : public MeshRefinement {
  public:

    /// Choose refinement rule
    static bool checkRule(Cell& cell, int no_marked_edges);

    /// Refine according to rule
    static void refine(Cell& cell, Mesh& mesh);

    // Friends
    friend class MeshRefinement;

  private:

    static bool checkRuleRegular   (Cell& cell, int no_marked_edges);
    static bool checkRuleIrregular1(Cell& cell, int no_marked_edges);
    static bool checkRuleIrregular2(Cell& cell, int no_marked_edges);

    static void refineNoRefine  (Cell& cell, Mesh& mesh);
    static void refineRegular   (Cell& cell, Mesh& mesh);
    static void refineIrregular1(Cell& cell, Mesh& mesh);
    static void refineIrregular2(Cell& cell, Mesh& mesh);

    static Cell& createCell(Node& n0, Node& n1, Node& n2, Mesh& mesh,
			    Cell& cell);
    
  };

}

#endif
