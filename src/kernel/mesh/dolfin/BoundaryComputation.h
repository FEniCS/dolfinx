// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-21
// Last changed: 2006-06-22

#ifndef __BOUNDARY_COMPUTATION_H
#define __BOUNDARY_COMPUTATION_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>

namespace dolfin
{

  class Mesh;
  class BoundaryMesh;

  /// This class implements provides a set of basic algorithms for
  /// the computation of boundaries.

  class BoundaryComputation
  {
  public:
    
    /// Compute the boundary of a given mesh
    static void computeBoundary(Mesh& mesh, BoundaryMesh& boundary,
				Array<uint>& icell);

  private:

    

  };

}

#endif
