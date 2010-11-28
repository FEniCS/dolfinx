// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-30
// Last changed: 2010-11-27

#ifndef __MESH_ORDERING_H
#define __MESH_ORDERING_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class Mesh;

  /// This class implements the ordering of mesh entities according to
  /// the UFC specification (see appendix of DOLFIN user manual).

  class MeshOrdering
  {
  public:

    /// Order mesh
    static void order(Mesh& mesh);

    /// Check if mesh is ordered
    static bool ordered(const Mesh& mesh);

  };

}

#endif
