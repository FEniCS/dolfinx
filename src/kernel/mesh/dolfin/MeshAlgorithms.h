// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-02

#ifndef __MESH_ALGORITHMS_H
#define __MESH_ALGORITHMS_H

#include <dolfin/constants.h>

namespace dolfin
{

  class NewMesh;

  /// This class implements a set of basic algorithms that automate
  /// the computation of mesh entities and extraction of boundaries.

  class MeshAlgorithms
  {
  public:

    /// Compute mesh entities of given topological dimension
    static void computeEntities(NewMesh& mesh, uint dim);

    /// Compute connectivity for given pair of topological dimensions
    static void computeConnectivity(NewMesh& mesh, uint d0, uint d1);

  private:

    /// Compute transpose of connectivity
    static void computeTranspose(NewMesh& mesh, uint d0, uint d1);

    /// Compute diagonal of connectitivy
    static void computeDiagonal(NewMesh& mesh, uint d);

    /// Generate connectivity from connectivity dim - 0
    static void generateConnectivity(NewMesh& mesh, uint d0, uint d1);

  };

}

#endif
