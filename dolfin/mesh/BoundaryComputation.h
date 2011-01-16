// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2006-06-21
// Last changed: 2010-03-02

#ifndef __BOUNDARY_COMPUTATION_H
#define __BOUNDARY_COMPUTATION_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  class BoundaryMesh;
  class Facet;
  class Mesh;
  template <class T> class MeshFunction;

  /// This class implements provides a set of basic algorithms for
  /// the computation of boundaries.

  class BoundaryComputation
  {
  public:

    /// Compute the exterior boundary of a given mesh
    static void compute_exterior_boundary(const Mesh& mesh,
                                          BoundaryMesh& boundary);

    /// Compute the interior boundary of a given mesh
    static void compute_interior_boundary(const Mesh& mesh,
                                          BoundaryMesh& boundary);

  private:

    /// Compute the boundary of a given mesh
    static void compute_boundary_common(const Mesh& mesh,
					                              BoundaryMesh& boundary,
					                              bool interior_boundary);

    /// Reorder vertices so facet is right-oriented w.r.t. facet normal
    static void reorder(std::vector<uint>& vertices, const Facet& facet);

  };

}

#endif
