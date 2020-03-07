// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <dolfinx/mesh/Mesh.h>

namespace dolfinx
{
namespace generation
{

/// A mesh consisting of a circular domain with quadratic geometry. This
/// class is useful for testing.

class UnitDiscMesh
{
public:
  /// Create mesh of unit disc for testing quadratic geometry
  /// @param[in] comm MPI communicator to build the mesh on
  /// @param[in] n Number of layers
  /// @param[in] ghost_mode Mesh ghosting mode
  static mesh::Mesh create(MPI_Comm comm, int n,
                           const mesh::GhostMode ghost_mode);
};
} // namespace generation
} // namespace dolfinx
