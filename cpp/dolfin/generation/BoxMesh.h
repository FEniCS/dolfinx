// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

namespace generation
{

/// Tetrahedral mesh of the 3D rectangular prism spanned by two
/// points p0 and p1. Given the number of cells (nx, ny, nz) in each
/// direction, the total number of tetrahedra will be 6*nx*ny*nz and
/// the total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

class BoxMesh
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
    if (cell_type == CellType::Type::tetrahedron)
      return build_tet(comm, p, n);
    else if (cell_type == CellType::Type::hexahedron)
      return build_hex(comm, n);
    else
    {
      dolfin_error("BoxMesh.h", "generate box mesh", "Wrong cell type '%d'",
                   cell_type);
    }

    // Will never reach this point
    return build_tet(comm, p, n);
    ;
  }

private:
  // Build mesh
  static Mesh build_tet(MPI_Comm comm, const std::array<Point, 2>& p,
                        std::array<std::size_t, 3> n);

  static Mesh build_hex(MPI_Comm comm, std::array<std::size_t, 3> n);
};
}
}
