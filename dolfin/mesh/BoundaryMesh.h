// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2006-10-16

#ifndef __BOUNDARY_MESH_H
#define __BOUNDARY_MESH_H

#include <dolfin/common/types.h>
#include "Mesh.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// A BoundaryMesh is a mesh over the boundary of some given mesh.

  class BoundaryMesh : public Mesh
  {
  public:

    /// Create an empty boundary mesh
    BoundaryMesh();

    /// Create boundary mesh from given mesh
    BoundaryMesh(Mesh& mesh);

    /// Create boundary mesh from given mesh and compute a pair
    /// of mappings from the vertices and cells of the boundary to
    /// the corresponding mesh entities in the original mesh
    BoundaryMesh(Mesh& mesh,
                 MeshFunction<uint>& vertex_map,
                 MeshFunction<uint>& cell_map);

    /// Destructor
    ~BoundaryMesh();

    /// Initialize boundary mesh from given mesh
    void init(Mesh& mesh);

    /// Initialize boundary mesh from given mesh, including a pair
    /// of mappings from the vertices and cells of the boundary to
    /// the corresponding mesh entities in the original mesh
    void init(Mesh& mesh,
              MeshFunction<uint>& vertex_map,
              MeshFunction<uint>& cell_map);

  };

}

#endif
