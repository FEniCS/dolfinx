// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-01

#ifndef __LOCAL_MESH_REFINEMENT_H
#define __LOCAL_MESH_REFINEMENT_H

#include <dolfin/constants.h>

namespace dolfin
{

  class Cell;
  class Mesh;
  class MeshHierarchy;

  /// This class implements local mesh refinement for different mesh types.

  class LocalMeshRefinement
  {
  public:

    /// Refine tetrahedral mesh by Bey algorithm 
    static void refineTetMesh(MeshHierarchy& mesh_hierarcy);
    
    /// Refine simplicial mesh locally by node insertion 
    static void refineSimplexByNodeInsertion(Mesh& mesh);

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

  };

}

#endif
