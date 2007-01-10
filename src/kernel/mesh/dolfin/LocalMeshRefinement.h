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

  /// This class implements local mesh refinement for different mesh types.

  class LocalMeshRefinement
  {
  public:

    /// Refine simplicial mesh locally by node insertion 
    static void refineSimplexMeshByBisection(Mesh& mesh, 
					     MeshFunction<bool>& cell_marker);


  };

}

#endif
