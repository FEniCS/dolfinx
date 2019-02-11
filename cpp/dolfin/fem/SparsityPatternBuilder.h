// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/la/SparsityPattern.h>

namespace dolfin
{
namespace la
{
class SparsityPattern;
}

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
  /// Iterate over cells and insert entries into sparsity pattern
  static void cells(la::SparsityPattern& pattern, const mesh::Mesh& mesh,
                    const std::array<const fem::GenericDofMap*, 2> dofmaps);

  /// Iterate over interior facets and insert entries into sparsity pattern
  static void
  interior_facets(la::SparsityPattern& pattern, const mesh::Mesh& mesh,
                  const std::array<const fem::GenericDofMap*, 2> dofmaps);

  /// Iterate over exterior facets and insert entries into sparsity pattern
  static void
  exterior_facets(la::SparsityPattern& pattern, const mesh::Mesh& mesh,
                  const std::array<const fem::GenericDofMap*, 2> dofmaps);
};
} // namespace fem
} // namespace dolfin