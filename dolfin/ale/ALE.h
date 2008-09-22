// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-08-11

#ifndef __ALE_H
#define __ALE_H

#include "ALEType.h"

namespace dolfin
{

  class Mesh;

  /// This class provides functionality useful for implementation of
  /// ALE (Arbitrary Lagrangian-Eulerian) methods, in particular
  /// moving the boundary vertices of a mesh and then interpolating
  /// the new coordinates for the interior vertices accordingly.

  class ALE
  {
  public:

    /// Move coordinates of mesh according to new boundary coordinates
    static void move(Mesh& mesh, Mesh& new_boundary, ALEType type=lagrange);

  };

}

#endif
