// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-07
// Last changed: 2006-06-16

#ifndef __UNIFORM_MESH_REFINEMENT_H
#define __UNIFORM_MESH_REFINEMENT_H

#include <dolfin/constants.h>

namespace dolfin
{

  class Mesh;

  /// This class implements uniform mesh refinement for different mesh types.

  class UniformMeshRefinement
  {
  public:

    /// Refine mesh uniformly according to mesh type
    static void refine(Mesh& mesh);

    /// Refine simplicial mesh uniformly
    static void refineSimplex(Mesh& mesh);

  };

}

#endif
