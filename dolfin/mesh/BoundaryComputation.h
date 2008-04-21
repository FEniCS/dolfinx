// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2008-04-21

#ifndef __BOUNDARY_COMPUTATION_H
#define __BOUNDARY_COMPUTATION_H

#include <dolfin/main/constants.h>
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

    /// Compute the boundary of a given mesh, including a pair of mappings
    /// from the vertices and cells of the boundary to the corresponding
    /// mesh entities in the original mesh
    static void computeBoundary(Mesh& mesh, BoundaryMesh& boundary,
                                MeshFunction<uint>& vertex_map,
                                MeshFunction<uint>& cell_map);

  private:

    /// Compute boundary and optionally the mappings if requested
    static void computeBoundaryCommon(Mesh& mesh, BoundaryMesh& boundary,
                                      MeshFunction<uint>* vertex_map,
                                      MeshFunction<uint>* cell_map);

    /// Reorder vertices so facet is right-oriented w.r.t. facet normal
    static void reorder(Array<uint>& vertices, Facet& facet);

  };

}

#endif
