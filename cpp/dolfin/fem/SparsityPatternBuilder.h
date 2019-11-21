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
class DofMap;

/// This class provides functions to compute the sparsity pattern
/// based on DOF maps

class SparsityPatternBuilder
{
public:
  /// Wrapper around accessor to local-global map for cell.
  /// 
  /// @param[in] i Either 0 or 1.
  /// @param[in] cell_index The cell index.
  /// @return    output of dofmaps[i]->cell_dofs(cell_index), where
  ///            dofmaps is an instance of std::array<
  ///              std::shared_ptr<const DofMap>, 2>
  using get_cell_dofs_function = std::function<
     Eigen::Array<PetscInt, Eigen::Dynamic, 1>(unsigned int, int)>;
          
  /// Iterate over cells and insert entries into sparsity pattern
  static void cells(la::SparsityPattern& pattern, const mesh::Mesh& mesh,
                    const get_cell_dofs_function & get_cell_dofs);

  /// Iterate over interior facets and insert entries into sparsity pattern
  static void interior_facets(la::SparsityPattern& pattern,
                              const mesh::Mesh& mesh,
                              const get_cell_dofs_function & get_cell_dofs);

  /// Iterate over exterior facets and insert entries into sparsity pattern
  static void exterior_facets(la::SparsityPattern& pattern,
                              const mesh::Mesh& mesh,
                              const get_cell_dofs_function & get_cell_dofs);
};
} // namespace fem
} // namespace dolfin
