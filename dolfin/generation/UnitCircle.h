// Copyright (C) 2005-2006 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Nuno Lopes 2008
//
// First added:  2005-12-02
// Last changed: 2006-08-19

#ifndef __UNIT_CIRCLE_H
#define __UNIT_CIRCLE_H

#include <dolfin/mesh/Mesh.h>

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
