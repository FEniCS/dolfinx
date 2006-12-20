// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-01

#ifndef __LOCAL_MESH_REFINEMENT_H
#define __LOCAL_MESH_REFINEMENT_H

#include <dolfin/constants.h>

namespace dolfin
{

  class Mesh;
  class Edge;

  /// This class implements local mesh refinement for different mesh types.

  class LocalMeshRefinement
  {
  public:

    /// Refine tetrahedral mesh by Bey algorithm 
    static void refineTetMesh(Mesh& mesh);

    /// Refine simplicial mesh locally by node insertion 
    static void refineSimplexByNodeInsertion(Mesh& mesh);

  };

}

#endif
