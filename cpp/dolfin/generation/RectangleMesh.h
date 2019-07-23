// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>
#include <string>

namespace dolfin
{

namespace generation
{

/// Triangular mesh of the 2D rectangle spanned by two points p0 and
/// p1. Given the number of cells (nx, ny) in each direction, the
/// total number of triangles will be 2*nx*ny and the total number
/// of vertices will be (nx + 1)*(ny + 1).

class RectangleMesh
{
public:
  /// @param    comm (MPI_Comm)
  ///         MPI communicator
  /// @param    p (std::array<_geometry::Point_, 2>)
  ///         Vertex points.
  /// @param    n (std::array<std::size_t, 2>)
  ///         Number of cells in each direction
  /// @param    cell_type (dolfin::CellType)
  ///         Cell type
  /// @param    diagonal (string)
  ///         Direction of diagonals: "left", "right", "left/right", "crossed"
  ///
  /// @code{.cpp}
  ///
  ///         // Mesh with 8 cells in each direction on the
  ///         // set [-1,2] x [-1,2]
  ///         geometry::Point p0(-1, -1);
  ///         geometry::Point p1(2, 2);
  ///         auto mesh = Rectanglemesh::Mesh::create(MPI_COMM_WORLD, {p0, p1},
  ///         {8,
  ///         8});
  /// @endcode
  static mesh::Mesh
    create(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
         std::array<std::size_t, 2> n, mesh::CellType cell_type,
         const mesh::GhostMode ghost_mode, std::string diagonal = "right");
};
} // namespace generation
} // namespace dolfin
