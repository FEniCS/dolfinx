// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/mesh/Mesh.h>

namespace dolfin
{
namespace generation
{

/// A mesh consisting of a circular domain with quadratic geometry.
/// This class is useful for testing.

class UnitDiscMesh
{
public:
  /// Create mesh of unit disc for testing quadratic geometry
  /// @param n
  ///   number of layers
  static mesh::Mesh create(MPI_Comm comm, std::size_t n);
};
}
}
