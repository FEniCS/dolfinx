// Copyright (C) 2007 Kristian B. Oelgaard
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <dolfinx/mesh/Mesh.h>

namespace dolfinx
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
  /// @param[in] comm MPI communicator to build the mesh on
  /// @param[in] n The number of cells.
  /// @param[in] x The end points
  /// @param[in] ghost_mode Ghosting mode
  /// @return A mesh
  static mesh::Mesh create(MPI_Comm comm, std::size_t n,
                           std::array<double, 2> x,
                           const mesh::GhostMode ghost_mode);
};
} // namespace generation
} // namespace dolfinx
