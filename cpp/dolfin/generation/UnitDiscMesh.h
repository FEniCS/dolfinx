// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
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
} // namespace dolfin
