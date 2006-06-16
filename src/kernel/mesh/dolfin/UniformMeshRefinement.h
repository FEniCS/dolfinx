// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-07
// Last changed: 2006-06-16

#ifndef __UNIFORM_MESH_REFINEMENT_H
#define __UNIFORM_MESH_REFINEMENT_H

#include <dolfin/constants.h>

namespace dolfin
{

  class NewMesh;

  /// This class implements uniform mesh refinement for different mesh types.

  class UniformMeshRefinement
  {
  public:

    /// Refine mesh uniformly according to mesh type
    static void refine(NewMesh& mesh);

    /// Refine simplicial mesh uniformly
    static void refineSimplex(NewMesh& mesh);



    /// Uniform mesh refinement for interval meshes
    static void refineInterval(NewMesh& mesh);

    /// Uniform mesh refinement for triangular meshes
    static void refineTriangle(NewMesh& mesh);

    /// Uniform mesh refinement for tetrahedral meshes
    static void refineTetrahedron(NewMesh& mesh);

  };

}

#endif
