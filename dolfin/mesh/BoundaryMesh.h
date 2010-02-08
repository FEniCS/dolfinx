// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2006-06-21
// Last changed: 2010-02-08

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

    /// Create (interior) boundary mesh from given mesh
    BoundaryMesh(const Mesh& mesh);

    /// Destructor
    ~BoundaryMesh();

    /// Initialize exterior boundary of given mesh
    void init_exterior_boundary(const Mesh& mesh);

    /// Initialize interior boundary of given mesh
    void init_interior_boundary(const Mesh& mesh);

  };

}

#endif
