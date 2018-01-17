// Copyright (C) 2007 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>

namespace dolfin
{

class GenericDofMap;
class Mesh;
class SparsityPattern;

/// This class provides functions to compute the sparsity pattern
/// based on DOF maps

class SparsityPatternBuilder
{
public:
  /// Build sparsity pattern for assembly of given bilinea form
  static void build(SparsityPattern& sparsity_pattern, const Mesh& mesh,
                    const std::array<const GenericDofMap*, 2> dofmaps,
                    bool cells, bool interior_facets, bool exterior_facets,
                    bool vertices, bool diagonal, bool init = true,
                    bool finalize = true);
};
}


