// Copyright (C) 2005-2011 Anders Logg
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
// First added:  2005-12-02
// Last changed: 2011-12-07

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
  /// *Arguments*
  ///     x0 (double)
  ///         :math:`x`-min.
  ///     y0 (double)
  ///         :math:`y`-min.
  ///     x1 (double)
  ///         :math:`x`-max.
  ///     y1 (double)
  ///         :math:`y`-max.
  ///     xn (double)
  ///         Number of cells in :math:`x`-direction.
  ///     yn (double)
  ///         Number of cells in :math:`y`-direction.
  ///     diagonal (string)
  ///         Direction of diagonals: "left", "right", "left/right", "crossed"
  ///
  /// *Example*
  ///     .. code-block:: c++
  ///
  ///         // Mesh with 6 cells in each direction on the
  ///         // set [-1,2] x [-1,2]
  ///         Box mesh(-1, -1, 2, 2, 6, 6;

  class Rectangle : public Mesh
  {
  public:

    Rectangle(double x0, double y0, double x1, double y1,
              uint nx, uint ny, std::string diagonal="right");

  };

}

#endif
