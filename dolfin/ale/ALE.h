// Copyright (C) 2008-2009 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2010-03-02

#ifndef __ALE_H
#define __ALE_H

#include "ALEType.h"

namespace dolfin
{

  class Mesh;
  class BoundaryMesh;
  class Function;

  /// This class provides functionality useful for implementation of
  /// ALE (Arbitrary Lagrangian-Eulerian) methods, in particular
  /// moving the boundary vertices of a mesh and then interpolating
  /// the new coordinates for the interior vertices accordingly.

  class ALE
  {
  public:

    /// Move coordinates of mesh according to new boundary coordinates
    static void move(Mesh& mesh, const BoundaryMesh& new_boundary, dolfin::ALEType method=lagrange);

    /// Move coordinates of mesh0 according to mesh1 with common global vertices
    static void move(Mesh& mesh0, const Mesh& mesh1, dolfin::ALEType method=lagrange);

    /// Move coordinates of mesh according to displacement function
    static void move(Mesh& mesh, const Function& displacement);

  };

}

#endif
