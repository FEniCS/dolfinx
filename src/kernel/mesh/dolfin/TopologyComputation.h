// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-21

#ifndef __TOPOLOGY_COMPUTATION_H
#define __TOPOLOGY_COMPUTATION_H

#include <dolfin/constants.h>

namespace dolfin
{

  class NewMesh;
  class MeshEntity;
  class MeshConnectivity;

  /// This class implements a set of basic algorithms that automate
  /// the computation of mesh entities and connectivity.

  class TopologyComputation
  {
  public:

    /// Compute mesh entities of given topological dimension
    static uint computeEntities(NewMesh& mesh, uint dim);

    /// Compute connectivity for given pair of topological dimensions
    static void computeConnectivity(NewMesh& mesh, uint d0, uint d1);

  private:

    /// Compute connectivity from transpose
    static void computeFromTranspose(NewMesh& mesh, uint d0, uint d1);

    /// Compute connectivity from intersection
    static void computeFromIntersection(NewMesh& mesh, uint d0, uint d1, uint d);

    /// Count how many of the given entities that are new
    static uint countEntities(NewMesh& mesh, MeshEntity& cell, 
			      uint** vertices, uint m, uint n, uint dim);

    /// Add entities that are new
    static void addEntities(NewMesh& mesh, MeshEntity& cell,
			    uint** vertices, uint m, uint n, uint dim,
			    MeshConnectivity& ce, MeshConnectivity& ev,
			    uint& current_entity);

    /// Check if mesh entity e0 contains mesh entity e1
    static bool contains(MeshEntity& e0, MeshEntity& e1);

    /// Check if array v0 contains array v1
    static bool contains(uint* v0, uint n0, uint* v1, uint n1);
    
  };

}

#endif
