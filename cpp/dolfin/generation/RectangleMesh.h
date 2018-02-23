// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <string>

namespace dolfin
{

/// Triangular mesh of the 2D rectangle spanned by two points p0 and
/// p1. Given the number of cells (nx, ny) in each direction, the
/// total number of triangles will be 2*nx*ny and the total number
/// of vertices will be (nx + 1)*(ny + 1).

class RectangleMesh : public Mesh
{
public:
  /// @param    comm (MPI_Comm)
  ///         MPI communicator
  /// @param    p (std::array<_Point_, 2>)
  ///         Vertex points.
  /// @param    n (std::array<std::size_t, 2>)
  ///         Number of cells in each direction
  /// @param    cell_type (dolfin::CellType::Type)
  ///         Cell type
  /// @param    diagonal (string)
  ///         Direction of diagonals: "left", "right", "left/right", "crossed"
  ///
  /// @code{.cpp}
  ///
  ///         // Mesh with 8 cells in each direction on the
  ///         // set [-1,2] x [-1,2]
  ///         Point p0(-1, -1);
  ///         Point p1(2, 2);
  ///         auto mesh = RectangleMesh::create(MPI_COMM_WORLD, {p0, p1}, {8,
  ///         8});
  /// @endcode
  static Mesh create(MPI_Comm comm, const std::array<Point, 2>& p,
                     std::array<std::size_t, 2> n, CellType::Type cell_type,
                     std::string diagonal = "right")
  {
    if (cell_type == CellType::Type::triangle)
      return build_tri(comm, p, n, diagonal);
    else if (cell_type == CellType::Type::quadrilateral)
      return build_quad(comm, p, n);
    else
    {
      dolfin_error("RectangleMesh.h", "generate rectangle mesh",
                   "Wrong cell type '%d'", cell_type);
    }

    // Will never reach this point
    return build_quad(comm, p, n);
  }

private:
  // Build mesh
  static Mesh build_tri(MPI_Comm comm, const std::array<Point, 2>& p,
                        std::array<std::size_t, 2> n,
                        std::string diagonal = "right");

  static Mesh build_quad(MPI_Comm comm, const std::array<Point, 2>& p,
                         std::array<std::size_t, 2> n);
};
}


