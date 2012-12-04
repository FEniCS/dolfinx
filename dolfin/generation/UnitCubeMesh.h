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
// Modified by Benjamin Kehlet 2012
//
// First added:  2005-12-02
// Last changed: 2012-11-09

#ifndef __UNIT_CUBE_MESH_H
#define __UNIT_CUBE_MESH_H

#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// Tetrahedral mesh of the 3D unit cube [0,1] x [0,1] x [0,1].
  /// Given the number of cells (nx, ny, nz) in each direction,
  /// the total number of tetrahedra will be 6*nx*ny*nz and the
  /// total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

  class UnitCubeMesh : public Mesh
  {
  public:

    /// Create a uniform finite element _Mesh_ over the unit cube
    /// [0,1] x [0,1] x [0,1].
    ///
    /// *Arguments*
    ///     nx (std::size_t)
    ///         Number of cells in :math:`x` direction.
    ///     ny (std::size_t)
    ///         Number of cells in :math:`y` direction.
    ///     nz (std::size_t)
    ///         Number of cells in :math:`z` direction.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         UnitCubeMesh mesh(32, 32, 32);
    ///
    UnitCubeMesh(std::size_t nx, std::size_t ny, std::size_t nz);

  };

}

#endif
