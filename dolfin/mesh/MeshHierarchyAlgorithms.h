// Copyright (C) 2006 Johan Hoffman
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
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
