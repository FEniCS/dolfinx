// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __UNIT_SQUARE_H
#define __UNIT_SQUARE_H

#include <dolfin/Mesh.h>

namespace dolfin
{

  /// This class represents a triangular mesh of the 2D unit square,
  /// i.e., (0,1) x (0,1). Given the number of cells (nx, ny) in each
  /// direction, the total number of triangles will be 2*nx*ny and the
  /// total number of nodes will be (nx + 1)*(ny + 1).

  class UnitSquare : public Mesh
  {
  public:

    UnitSquare(uint nx, uint ny);

  };
  
}

#endif
