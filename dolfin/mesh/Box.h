// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes, 2008.
//
// First added:  2005-12-02
// Last changed: 2006-08-07

#ifndef __BOX_H
#define __BOX_H

#include "Mesh.h"

namespace dolfin
{

  /// Tetrahedral mesh of the 3D  rectangular prysm (a,b) x (c,d) x (e,f).
  /// Given the number of cells (nx, ny, nz) in each direction,
  /// the total number of tetrahedra will be 6*nx*ny*nz and the
  /// total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

  class Box : public Mesh
  {
  public:

    Box(real a, real b, real c, real d, real e, real f, uint nx, uint ny, 
        uint nz);

  };
  
}

#endif
