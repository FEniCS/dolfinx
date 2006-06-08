// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-08

#ifndef __MESH_ALGORITHMS_H
#define __MESH_ALGORITHMS_H

#include <dolfin/Array.h>
#include <dolfin/constants.h>

namespace dolfin
{

  class NewMesh;
  class MeshEntity;
  class MeshConnectivity;

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

    /// Count how many of the given entities that are new
    static uint countEntities(NewMesh& mesh, MeshEntity& cell, Array<Array<uint> >& entities);

    /// Add entities that are new
    static void addEntities(NewMesh& mesh, MeshEntity& cell, Array<Array<uint> >& entities, 
			    MeshConnectivity& connectivity, uint& current_entity);

    /// Check if entity contains all given vertices
    static bool containsVertices(MeshEntity& entity, Array<uint>& vertices);
    
  };

}

#endif
