// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-21
// Last changed: 2006-06-22

#ifndef __BOUNDARY_COMPUTATION_H
#define __BOUNDARY_COMPUTATION_H

#include <dolfin/constants.h>
#include <dolfin/MeshFunction.h>

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

  };

}

#endif
