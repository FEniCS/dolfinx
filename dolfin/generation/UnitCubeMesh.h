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
// Modified by Benjamin Kehlet 2012
// Modified by Mikael Mortensen 2014
//
// First added:  2005-12-02
// Last changed: 2015-06-15

#ifndef __UNIT_CUBE_MESH_H
#define __UNIT_CUBE_MESH_H

#include <cstddef>
#include <dolfin/common/MPI.h>
#include "BoxMesh.h"

namespace dolfin
{

  /// Tetrahedral mesh of the 3D unit cube [0,1] x [0,1] x [0,1].
  /// Given the number of cells (nx, ny, nz) in each direction,
  /// the total number of tetrahedra will be 6*nx*ny*nz and the
  /// total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

  class UnitCubeMesh : public BoxMesh
  {
  public:

    /// Create a uniform finite element _Mesh_ over the unit cube
    /// [0,1] x [0,1] x [0,1].
    ///
    /// @param    nx (std::size_t)
    ///         Number of cells in :math:`x` direction.
    /// @param    ny (std::size_t)
    ///         Number of cells in :math:`y` direction.
    /// @param    nz (std::size_t)
    ///         Number of cells in :math:`z` direction.
    ///
    /// @code{.cpp}
    ///
    ///         UnitCubeMesh mesh(32, 32, 32);
    /// @endcode
    UnitCubeMesh(std::size_t nx, std::size_t ny, std::size_t nz)
      : UnitCubeMesh(MPI_COMM_WORLD, nx, ny, nz) {}

    /// Create a uniform finite element _Mesh_ over the unit cube
    /// [0,1] x [0,1] x [0,1].
    ///
    /// @param    comm (MPI_Comm)
    ///         MPI communicator
    /// @param    nx (std::size_t)
    ///         Number of cells in :math:`x` direction.
    /// @param    ny (std::size_t)
    ///         Number of cells in :math:`y` direction.
    /// @param    nz (std::size_t)
    ///         Number of cells in :math:`z` direction.
    ///
    /// @code{.cpp}
    ///         UnitCubeMesh mesh(MPI_COMM_WORLD, 32, 32, 32);
    /// @endcode
    UnitCubeMesh(MPI_Comm comm, std::size_t nx, std::size_t ny, std::size_t nz)
      : BoxMesh(comm, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0), nx, ny, nz) {}

  };

}

#endif
