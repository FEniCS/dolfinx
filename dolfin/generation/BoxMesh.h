// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
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

#ifndef __BOX_H
#define __BOX_H

#include <array>
#include <cstddef>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

/// Tetrahedral mesh of the 3D rectangular prism spanned by two
/// points p0 and p1. Given the number of cells (nx, ny, nz) in each
/// direction, the total number of tetrahedra will be 6*nx*ny*nz and
/// the total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

class BoxMesh : public Mesh
{
public:
  /// Create a uniform finite element _Mesh_ over the rectangular
  /// prism spanned by the two _Point_s p0 and p1. The order of the
  /// two points is not important in terms of minimum and maximum
  /// coordinates.
  ///
  /// @param comm (MPI_Comm)
  ///         MPI communicator
  /// @param p (std::array<_Point_, 2>)
  ///         Points of box.
  /// @param n (std::array<double, 3> )
  ///         Number of cells in each direction.
  /// @param cell_type
  ///         Tetrahedron or hexahedron
  ///
  /// @code{.cpp}
  ///         // Mesh with 8 cells in each direction on the
  ///         // set [-1,2] x [-1,2] x [-1,2].
  ///         Point p0(-1, -1, -1);
  ///         Point p1(2, 2, 2);
  ///         auto mesh = BoxMesh::create({p0, p1}, {8, 8, 8});
  /// @endcode
  static Mesh create(MPI_Comm comm, const std::array<Point, 2>& p,
                     std::array<std::size_t, 3> n, CellType::Type cell_type)
  {
    Mesh mesh(comm);
    if (cell_type == CellType::Type::tetrahedron)
      build_tet(mesh, p, n);
    else if (cell_type == CellType::Type::hexahedron)
      build_hex(mesh, n);
    else
    {
      dolfin_error("BoxMesh.h", "generate box mesh", "Wrong cell type '%d'",
                   cell_type);
    }

    return mesh;
  }

private:
  // Build mesh
  static void build_tet(Mesh& mesh, const std::array<Point, 2>& p,
                        std::array<std::size_t, 3> n);

  static void build_hex(Mesh& mesh, std::array<std::size_t, 3> n);
};
}

#endif
