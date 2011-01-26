// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-02
// Last changed: 2009-09-29

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
  /// std::string diagonal ("left", "right", "right/left", "left/right", or "crossed")
  /// indicates the direction of the diagonals.

  class Rectangle : public Mesh
  {
  public:

    Rectangle(double x0, double y0, double x1, double y1,
              uint nx, uint ny, std::string diagonal = "right");

  };

}

#endif
