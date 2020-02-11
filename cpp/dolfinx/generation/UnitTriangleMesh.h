// Copyright (C) 2010 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/mesh/Mesh.h>

namespace dolfinx
{
namespace generation
{

/// A mesh consisting of a single triangle with vertices at
///
///   (0, 0)
///   (1, 0)
///   (0, 1)
///
/// This class is useful for testing.

class UnitTriangleMesh
{
public:
  /// Create mesh of unit triangle
  static mesh::Mesh create();
};
} // namespace generation
} // namespace dolfinx
