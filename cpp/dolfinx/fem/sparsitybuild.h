// Copyright (C) 2007-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include <array>
#include <cstdint>
#include <dolfinx/la/SparsityPattern.h>
#include <functional>
#include <span>

// namespace dolfinx::la
// {
// class SparsityPattern;
// }

namespace dolfinx::fem
{
// class DofMap;

/// Support for building sparsity patterns from degree-of-freedom maps.
namespace sparsitybuild
{
/// @brief Iterate over cells and insert entries into sparsity pattern.
///
/// Inserts the rectangular blocks of indices `dofmap[0][cells[0][i]] x
/// dofmap[1][cells[1][i]]` into the sparsity pattern, i.e. entries
/// `(dofmap[0][cells[0][i]][k0], dofmap[0][cells[0][i]][k1])` will
/// appear in the sparsity pattern.
///
/// @param pattern Sparsity pattern to insert into.
/// @param cells0 Lists of cells to iterate over. `cells[0]` and
/// `cells[1]` must have the same size.
/// @param cells1 Lists of cells to iterate over. `cells[0]` and
/// `cells[1]` must have the same size.
/// @param dofmaps Dofmaps to used in building the sparsity pattern.
/// @note The sparsity pattern is not finalised.
template <typename R>
void cells(la::SparsityPattern& pattern, R&& cells0, R&& cells1,
           std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  assert(cells0.size() == cells1.size());
  auto [map0, map1] = dofmaps;
  for (std::size_t i = 0; i < cells0.size(); ++i)
  {
    pattern.insert(map0.get().cell_dofs(cells0[i]),
                   map1.get().cell_dofs(cells1[i]));
  }
}

/// @brief Iterate over interior facets and insert entries into sparsity
/// pattern.
///
/// Inserts the rectangular blocks of indices `[dofmap[0][cell0],
/// dofmap[0][cell1]] x [dofmap[1][cell0] + dofmap[1][cell1]]` where
/// `cell0` and `cell1` are the two cells attached to a facet.
///
/// @param[in,out] pattern Sparsity pattern to insert into.
/// @param[in] cells Cells to index into each dofmap. `cells[i]` is a
/// list of `(cell0, cell1)` pairs for each interior facet to index into
/// `dofmap[i]`. `cells[0]` and `cells[1]` must have the same size.
/// @param[in] dofmaps Dofmaps to use in building the sparsity pattern.
///
/// @note The sparsity pattern is not finalised.
void interior_facets(
    la::SparsityPattern& pattern,
    std::array<std::span<const std::int32_t>, 2> cells,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps);

} // namespace sparsitybuild
} // namespace dolfinx::fem
