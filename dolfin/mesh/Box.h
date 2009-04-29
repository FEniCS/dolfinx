// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes, 2008.
//
// First added:  2005-12-02
// Last changed: 2009-02-11

#ifndef __BOX_H
#define __BOX_H

#include "Mesh.h"

namespace dolfin
{

  /// Tetrahedral mesh of the 3D  rectangular prism (x0, y0) x (x1, y1) x (x2, y2).
  /// Given the number of cells (nx, ny, nz) in each direction,
  /// the total number of tetrahedra will be 6*nx*ny*nz and the
  /// total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

  class Box : public Mesh
  {
  public:

    Box(double x0, double y0, double z0,
        double x1, double y1, double z1,
        uint nx, uint ny, uint nz);

  };

}

#endif
