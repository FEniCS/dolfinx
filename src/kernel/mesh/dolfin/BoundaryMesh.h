// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-21
// Last changed: 2006-06-22

#ifndef __BOUNDARY_MESH_H
#define __BOUNDARY_MESH_H

#include <dolfin/constants.h>
#include <dolfin/NewMesh.h>

namespace dolfin
{

  /// A BoundaryMesh is a mesh over the boundary of some given mesh.

  class BoundaryMesh : public NewMesh
  {
  public:

    /// Create an empty boundary mesh
    BoundaryMesh();

    /// Create boundary mesh from given mesh
    BoundaryMesh(NewMesh& mesh);

    /// Destructor
    ~BoundaryMesh();

    /// Initialize boundary mesh from given mesh
    void init(NewMesh& mesh);

  };

}

#endif
