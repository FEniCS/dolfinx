// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2008-05-28

#ifndef __BOUNDARY_COMPUTATION_H
#define __BOUNDARY_COMPUTATION_H

#include <dolfin/common/types.h>
#include "MeshFunction.h"

namespace dolfin
{

  class Mesh;
  class Facet;
  class BoundaryMesh;

  /// This class implements provides a set of basic algorithms for
  /// the computation of boundaries.

  class BoundaryComputation
  {
  public:
    
    /// Compute the boundary of a given mesh
    static void computeBoundary(Mesh& mesh, BoundaryMesh& boundary);

  private:

    /// Reorder vertices so facet is right-oriented w.r.t. facet normal
    static void reorder(Array<uint>& vertices, Facet& facet);

  };

}

#endif
