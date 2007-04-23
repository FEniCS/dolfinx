// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-30
// Last changed: 2007-01-30

#ifndef __MESH_ORDERING_H
#define __MESH_ORDERING_H

#include <dolfin/constants.h>

namespace dolfin
{

  class Mesh;

  /// This class implements the ordering of mesh entities according to
  /// the UFC specification (see appendix of DOLFIN user manual).

  class MeshOrdering
  {
  public:

    static void order(Mesh& mesh);

  };

}

#endif
