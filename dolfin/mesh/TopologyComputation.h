// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-02
// Last changed: 2010-11-25

#ifndef __TOPOLOGY_COMPUTATION_H
#define __TOPOLOGY_COMPUTATION_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class Mesh;
  class MeshEntity;
  class MeshConnectivity;

  /// This class implements a set of basic algorithms that automate
  /// the computation of mesh entities and connectivity.

  class TopologyComputation
  {
  public:

    /// Compute mesh entities of given topological dimension
    static uint compute_entities(Mesh& mesh, uint dim);

    /// Compute connectivity for given pair of topological dimensions
    static void compute_connectivity(Mesh& mesh, uint d0, uint d1);

  private:

    /// Compute connectivity from transpose
    static void compute_from_transpose(Mesh& mesh, uint d0, uint d1);

    /// Compute connectivity from intersection
    static void compute_from_intersection(Mesh& mesh, uint d0, uint d1, uint d);

    /// Count how many of the given entities that are new
    static uint count_entities(Mesh& mesh, MeshEntity& cell,
			      uint** vertices, uint m, uint n, uint dim);

    /// Add entities that are new
    static void add_entities(Mesh& mesh, MeshEntity& cell,
			    uint** vertices, uint m, uint n, uint dim,
			    MeshConnectivity& ce, MeshConnectivity& ev,
			    uint& current_entity);

    /// Check if mesh entity e0 contains mesh entity e1
    static bool contains(MeshEntity& e0, MeshEntity& e1);

    /// Check if array v0 contains array v1
    static bool contains(const uint* v0, uint n0, const uint* v1, uint n1);

  };

}

#endif
