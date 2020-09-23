// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfinx/la/SparsityPattern.h>

namespace dolfinx
{
namespace la
{
class SparsityPattern;
}

namespace mesh
{
class Topology;
}

namespace fem
{
class DofMap;

/// Functions to compute the sparsity pattern based on DOF maps

namespace SparsityPatternBuilder
{

/// Iterate over cells and insert entries into sparsity pattern
void cells(la::SparsityPattern& pattern, const mesh::Topology& topology,
           const std::array<const fem::DofMap*, 2> dofmaps);

/// Iterate over interior facets and insert entries into sparsity pattern
void interior_facets(la::SparsityPattern& pattern,
                     const mesh::Topology& topology,
                     const std::array<const fem::DofMap*, 2> dofmaps);

/// Iterate over exterior facets and insert entries into sparsity pattern
void exterior_facets(la::SparsityPattern& pattern,
                     const mesh::Topology& topology,
                     const std::array<const fem::DofMap*, 2> dofmaps);

} // namespace SparsityPatternBuilder
} // namespace fem
} // namespace dolfinx
