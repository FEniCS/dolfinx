// Copyright (C) 2012 Benjamin Kehlet
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
// First added:  2012-11-09
// Last changed: 2012-11-09

#ifndef __UNIT_SQUARE_H
#define __UNIT_SQUARE_H

#include "UnitSquareMesh.h"

namespace dolfin
{

  /// Triangular mesh of the 2D unit square [0,1] x [0,1].
  /// Given the number of cells (nx, ny) in each direction,
  /// the total number of triangles will be 2*nx*ny and the
  /// total number of vertices will be (nx + 1)*(ny + 1).
  ///
  /// std::string diagonal ("left", "right", "right/left", "left/right",
  /// or "crossed") indicates the direction of the diagonals.
  ///
  /// This class is deprecated. Use _UnitSquareMesh_.
  class UnitSquare : public UnitSquareMesh
  {
  public:

    /// Create a uniform finite element _Mesh_ over the unit square
    /// [0,1] x [0,1].
    ///
    /// *Arguments*
    ///     nx (std::size_t)
    ///         Number of cells in horizontal direction.
    ///     ny (std::size_t)
    ///         Number of cells in vertical direction.
    ///     diagonal (std::string)
    ///         Optional argument: A std::string indicating
    ///         the direction of the diagonals.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         UnitSquare mesh1(32, 32);
    ///         UnitSquare mesh2(32, 32, "crossed");
    ///
    UnitSquare(std::size_t nx, std::size_t ny,
               std::string diagonal="right")
      : UnitSquareMesh(nx, ny, diagonal)
    {
      warning("UnitSquare is deprecated. Use UnitSquareMesh.");
    }

  };

}

#endif
