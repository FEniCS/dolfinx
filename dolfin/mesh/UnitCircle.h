// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes 2008
//
// First added:  2005-12-02
// Last changed: 2006-08-19

#ifndef __UNIT_CIRCLE_H
#define __UNIT_CIRCLE_H

#include "Mesh.h"

namespace dolfin
{

  /// Triangular mesh of the 2D unit circle. 
  /// Given the number of cells (nx, ny) in each direction,
  /// the total number of triangles will be 2*nx*ny and the
  /// total number of vertices will be (nx + 1)*(ny + 1).

  /// The Type is an enumerater taking values in {left, right or
  /// crisscross} indicating the direction of the diagonals for
  /// left/right or both == crisscross. The default is right.

  class UnitCircle : public Mesh
  {
  public:

    enum Type {right, left, crisscross};
    enum Transformation {maxn, sumn, rotsumn};

    UnitCircle(uint nx, Type type = crisscross, 
               Transformation transformation = rotsumn);

  private:

    real transformx(real x, real y, Transformation transformation);

    real transformy(real x, real y, Transformation transformation);

    inline real max(real x, real y)
    { return ((x>y) ? x : y); };
  };
  
}

#endif
