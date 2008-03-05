// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-09

#ifndef __MESH_HIERARCHY_ALGORITHMS_H
#define __MESH_HIERARCHY_ALGORITHMS_H

#include <dolfin/main/constants.h>

namespace dolfin
{

  class Cell;
  class Mesh;
  class MeshHierarchy;

  /// This class implements algorithms on a MeshHierarchy 

  class MeshHierarchyAlgorithms
  {
  public:

    /// Refine tetrahedral mesh by Bey algorithm 
    static void refineTetMesh(MeshHierarchy& mesh_hierarcy);
    
  private:

    /// This function implements the "EvaluateMarks" subroutine by Bey 
    void evaluateMarks(Mesh& mesh); 
      
    /// This function implements the "CloseGrid" subroutine by Bey 
    void closeMesh(Mesh& mesh); 

    /// This function implements the "CloseElement" subroutine by Bey 
    void closeCell(Cell& cell); 

    /// This function implements the "UnrefineGrid" subroutine by Bey 
    void unrefineMesh(MeshHierarchy& mesh, uint k);

    /// This function implements the "RefineGrid" subroutine by Bey 
    void refineMesh(MeshHierarchy& mesh, uint k);

  MeshFunction<uint>* cell_marker; 
  MeshFunction<uint>* cell_state; 

  };

}

#endif
