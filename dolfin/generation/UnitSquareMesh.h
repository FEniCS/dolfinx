// Copyright (C) 2005-2015 Anders Logg
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
// Modified by Mikael Mortensen, 2014
//
// First added:  2005-12-02
// Last changed: 2015-06-15

#ifndef __UNIT_SQUARE_MESH_H
#define __UNIT_SQUARE_MESH_H

#include <string>
#include "RectangleMesh.h"

namespace dolfin
{

  /// Triangular mesh of the 2D unit square [0,1] x [0,1].
  /// Given the number of cells (nx, ny) in each direction,
  /// the total number of triangles will be 2*nx*ny and the
  /// total number of vertices will be (nx + 1)*(ny + 1).
  ///
  /// std::string diagonal ("left", "right", "right/left", "left/right",
  /// or "crossed") indicates the direction of the diagonals.

  class UnitSquareMesh : public RectangleMesh
  {
  public:

    /// Create a uniform finite element _Mesh_ over the unit square
    /// [0,1] x [0,1].
    ///
    /// @param    nx (std::size_t)
    ///         Number of cells in horizontal direction.
    /// @param    ny (std::size_t)
    ///         Number of cells in vertical direction.
    /// @param    diagonal (std::string)
    ///         Optional argument: A std::string indicating
    ///         the direction of the diagonals.
    ///
    /// @code{.cpp}
    ///
    ///         UnitSquareMesh mesh1(32, 32);
    ///         UnitSquareMesh mesh2(32, 32, "crossed");
    /// @endcode
    UnitSquareMesh(std::size_t nx, std::size_t ny, std::string diagonal="right")
      : UnitSquareMesh(MPI_COMM_WORLD, nx, ny, diagonal) {}

    /// Create a uniform finite element _Mesh_ over the unit square
    /// [0,1] x [0,1].
    ///
    /// @param    comm (MPI_Comm)
    ///         MPI communicator
    /// @param    nx (std::size_t)
    ///         Number of cells in horizontal direction.
    /// @param    ny (std::size_t)
    ///         Number of cells in vertical direction.
    /// @param    diagonal (std::string)
    ///         Optional argument: A std::string indicating
    ///         the direction of the diagonals.
    ///
    /// @code{.cpp}
    ///
    ///         UnitSquareMesh mesh1(MPI_COMM_WORLD, 32, 32);
    ///         UnitSquareMesh mesh2(MPI_COMM_WORLD, 32, 32, "crossed");
    /// @endcode
    UnitSquareMesh(MPI_Comm comm, std::size_t nx, std::size_t ny,
                   std::string diagonal="right")
      : RectangleMesh(comm, Point(0.0, 0.0), Point(1.0, 1.0), nx, ny, diagonal) {}

  };

}

#endif
