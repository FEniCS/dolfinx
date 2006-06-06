// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-06

#ifndef __MESH_ALGORITHMS_H
#define __MESH_ALGORITHMS_H

#include <dolfin/Array.h>
#include <dolfin/constants.h>

namespace dolfin
{

  class NewMesh;
  class MeshEntity;

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

    /// Compute connectivity from transpose
    static void computeFromTranspose(NewMesh& mesh, uint d0, uint d1);

    /// Compute connectivity from intersection
    static void computeFromIntersection(NewMesh& mesh, uint d0, uint d1, uint d);

    /// Count the number of entities of topological dimension dim
    static uint countEntities(NewMesh& mesh, uint dim);

    /// Add entities of topological dimension dim
    static void addEntities(NewMesh& mesh, uint dim, uint num_entities);

    /// Check if entity contains all given vertices
    static bool containsVertices(MeshEntity& entity, Array<uint>& vertices);
    
  };

}

#endif
