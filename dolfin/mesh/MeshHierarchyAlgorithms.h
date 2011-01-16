// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-09

#ifndef __MESH_HIERARCHY_ALGORITHMS_H
#define __MESH_HIERARCHY_ALGORITHMS_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class Cell;
  class Mesh;
  class MeshHierarchy;
  template <class T> class MeshFunction;

  /// This class implements algorithms on a MeshHierarchy

  class MeshHierarchyAlgorithms
  {
  public:

    /// Refine tetrahedral mesh by Bey algorithm
    static void refineTetMesh(MeshHierarchy& mesh_hierarcy);

  private:

    /// This function implements the "EvaluateMarks" subroutine by Bey
    void evaluate_marks(Mesh& mesh);

    /// This function implements the "CloseGrid" subroutine by Bey
    void close_mesh(Mesh& mesh);

    /// This function implements the "CloseElement" subroutine by Bey
    void close_cell(Cell& cell);

    /// This function implements the "UnrefineGrid" subroutine by Bey
    void unrefine_mesh(MeshHierarchy& mesh, uint k);

    /// This function implements the "RefineGrid" subroutine by Bey
    void refine_mesh(MeshHierarchy& mesh, uint k);

    MeshFunction<uint>* cell_marker;
    MeshFunction<uint>* cell_state;

  };

}

#endif
