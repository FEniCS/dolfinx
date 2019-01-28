// Copyright (C) 2007 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/MPI.h>
#include <dolfin/la/SparsityPattern.h>

namespace dolfin
{
namespace mesh
{
class Mesh;
}

namespace fem
{
class GenericDofMap;

/// This class provides functions to compute the sparsity pattern
/// based on DOF maps

class SparsityPatternBuilder
{
public:
  // FIXME: Simplify
  /// Build sparsity pattern for assembly of given bilinear form
  static la::SparsityPattern
  build(MPI_Comm comm, const mesh::Mesh& mesh,
        const std::array<const fem::GenericDofMap*, 2> dofmaps, bool cells,
        bool interior_facets, bool exterior_facets);
};
} // namespace fem
} // namespace dolfin