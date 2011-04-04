// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2010
//
// First added:  2006-06-07
// Last changed: 2010-02-26

#ifndef __UNIFORM_MESH_REFINEMENT_H
#define __UNIFORM_MESH_REFINEMENT_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class Mesh;

  /// This class implements uniform mesh refinement.

  class UniformMeshRefinement
  {
  public:

    /// Refine mesh uniformly
    static void refine(Mesh& refined_mesh, const Mesh& mesh);

  };

}

#endif
