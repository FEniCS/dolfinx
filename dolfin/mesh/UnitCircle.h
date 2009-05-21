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

  /// std::string diagonal ("left", "right" or "crossed") indicates the 
  /// direction of the diagonals.

  /// std:string transformation ("maxn", "sumn" or "rotsumn")

  class UnitCircle : public Mesh
  {
  public:

    UnitCircle(uint nx, std::string diagonal = "crossed",
               std::string transformation = "rotsumn");

  private:

    void transform(double* x_trans, double x, double y, std::string transformation);

    double transformx(double x, double y, std::string transformation);
    double transformy(double x, double y, std::string transformation);

    inline double max(double x, double y)
    { return ((x>y) ? x : y); };
  };

}

#endif
