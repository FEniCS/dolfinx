// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-02
// Last changed: 2009-02-11

#ifndef __RECTANGLE_H
#define __RECTANGLE_H

#include "Mesh.h"

namespace dolfin
{

  /// Triangular mesh of the 2D rectangle (x0, y0) x (x1, y1).
  /// Given the number of cells (nx, ny) in each direction,
  /// the total number of triangles will be 2*nx*ny and the
  /// total number of vertices will be (nx + 1)*(ny + 1).
  ///
  /// The Type is an enumerater taking values in {left, right or
  /// crisscross} indicating the direction of the diagonals for
  /// left/right or both == crisscross. The default is right.

  class Rectangle : public Mesh
  {
  public:

    enum Type {right, left, crisscross};

    Rectangle(double x0, double y0, double x1, double y1,
              uint nx, uint ny, Type type=right);

  };

}

#endif
