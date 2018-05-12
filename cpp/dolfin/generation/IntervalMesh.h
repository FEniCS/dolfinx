// Copyright (C) 2007 Kristian B. Oelgaard
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{
namespace generation
{

/// Interval mesh of the 1D line [a,b].  Given the number of cells
/// (n) in the axial direction, the total number of intervals will
/// be n and the total number of vertices will be (n + 1).

class IntervalMesh
{
public:
  /// Factory
  ///
  /// @param    comm (MPI_Comm)
  ///         MPI communicator
  /// @param    n (std::size_t)
  ///         The number of cells.
  /// @param    x (std::array<double, 2>)
  ///         The end points
  ///
  /// @code{.cpp}
  ///
  ///         // Create a mesh of 25 cells in the interval [-1,1]
  ///         IntervalMesh mesh(MPI_COMM_WORLD, 25, -1.0, 1.0);
  /// @endcode
  static mesh::Mesh create(MPI_Comm comm, std::size_t n,
                           std::array<double, 2> x,
                           const mesh::GhostMode ghost_mode)
  {
    return build(comm, n, x, ghost_mode);
  }

private:
  // Build mesh
  static mesh::Mesh build(MPI_Comm comm, std::size_t n,
                          std::array<double, 2> x,
                          const mesh::GhostMode ghost_mode);
};
}
}