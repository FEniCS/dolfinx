// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-02
// Last changed: 2006-08-07

#ifndef __UNIT_SQUARE_H
#define __UNIT_SQUARE_H

#include "Mesh.h"

namespace dolfin
{

  /// Triangular mesh of the 2D unit square (0,1) x (0,1). 
  /// Given the number of cells (nx, ny) in each direction,
  /// the total number of triangles will be 2*nx*ny and the
  /// total number of vertices will be (nx + 1)*(ny + 1).

  /// The Type is an enumerater taking values in {left, right or
  /// crisscross} indicating the direction of the diagonals for
  /// left/right or both == crisscross. The default is right.

  class UnitSquare : public Mesh
  {
  public:
    enum Type {right, left, crisscross};
    
    UnitSquare(uint nx, uint ny, Type type = right);

  };
  
}

#endif
