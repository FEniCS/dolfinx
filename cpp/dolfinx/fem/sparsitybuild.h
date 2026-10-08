// Copyright (C) 2007-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "ElementDofLayout.h"
#include <array>
#include <cstdint>
#include <dolfinx/la/SparsityPattern.h>
#include <functional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::fem
{
class DofMap;

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
/// @param cells Lists of cells to iterate over. `cells[0]` and
/// `cells[1]` must have the same size.
/// @param dofmaps Dofmaps to used in building the sparsity pattern.
/// @note The sparsity pattern is not finalised.
template <std::ranges::input_range R0, std::ranges::input_range R1>
void cells(la::SparsityPattern& pattern, const std::pair<R0, R1>& cells,
           std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  assert(cells.first.size() == cells.second.size());
  const DofMap& map0 = dofmaps[0].get();
  const DofMap& map1 = dofmaps[1].get();

  // Reserve entries for all cell-wise outer products.
  if constexpr (std::ranges::sized_range<R0> and std::ranges::sized_range<R1>)
  {
    if (std::size_t num_cells = std::ranges::size(cells.first); num_cells > 0)
    {
      std::size_t n0 = map0.cell_dofs(*cells.first.begin()).size();
      std::size_t n1 = map1.cell_dofs(*cells.second.begin()).size();
      pattern.reserve_blocks(num_cells, num_cells * n0, num_cells * n1);
    }
  }

  for (auto cell0 = cells.first.begin(), cell1 = cells.second.begin();
       cell0 != cells.first.end() and cell1 != cells.second.end();
       ++cell0, ++cell1)
  {
    pattern.insert(map0.cell_dofs(*cell0), map1.cell_dofs(*cell1));
  }
}

/// @brief Element-matrix blocks of an entity-closure stencil.
///
/// @param[in] layout1 Dof layout of the space of the rows.
/// @param[in] layout0 Dof layout of the space of the columns.
/// @return For each reference-cell entity carrying `layout1`
/// degrees-of-freedom, the cell-local `layout1` degrees-of-freedom on
/// the entity and the cell-local `layout0` degrees-of-freedom on its
/// closure. The spans point into the layouts, which must outlive the
/// return value.
std::vector<std::pair<std::span<const int>, std::span<const int>>>
entity_closure_blocks(const ElementDofLayout& layout1,
                      const ElementDofLayout& layout0);

/// @brief Iterate over cells and insert the entity-closure blocks into
/// a sparsity pattern.
///
/// Inserts, for each mesh entity of each cell, the `dofmaps[0]`
/// degrees-of-freedom on the entity against the `dofmaps[1]`
/// degrees-of-freedom on the closure of that entity. This is narrower
/// than sparsitybuild::cells, and is the sparsity of an operator whose
/// `dofmaps[0]` degrees-of-freedom are moments over the entity they are
/// attached to, since such a moment sees only `dofmaps[1]` restricted
/// to that entity. fem::discrete_gradient and fem::discrete_curl insert
/// into exactly these blocks.
///
/// Entries that the operator evaluates to zero are part of the
/// structure and are included; a consumer that reads the sparsity
/// rather than the values, such as PETSc's PCBDDC Nedelec support,
/// requires them.
///
/// @param pattern Sparsity pattern to insert into.
/// @param cells Cells to iterate over.
/// @param dofmaps Dofmaps used in building the sparsity pattern,
/// `dofmaps[0]` for the rows and `dofmaps[1]` for the columns.
/// @note The sparsity pattern is not finalised.
void entity_closure(
    la::SparsityPattern& pattern, std::span<const std::int32_t> cells,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps);

/// @brief Iterate over interior facets and insert entries into sparsity
/// pattern.
///
/// Inserts the rectangular block of indices `[dofmap[0][cell0],
/// dofmap[0][cell1]] x [dofmap[1][cell0], dofmap[1][cell1]]` where
/// `cell0` and `cell1` are the two cells attached to a facet.
///
/// @param[in,out] pattern Sparsity pattern to insert into.
/// @param[in] cells Cells to index into each dofmap. `cells[i]` is a
/// list of `(cell0, cell1)` pairs for each interior facet to index into
/// `dofmap[i]`. `cells[0]` and `cells[1]` must have the same size. A
/// negative cell index means no cell exists on that side (e.g. an
/// interface between two domains); no entries are inserted for it.
/// @param[in] dofmaps Dofmaps to use in building the sparsity pattern.
///
/// @note The sparsity pattern is not finalised.
void interior_facets(
    la::SparsityPattern& pattern,
    std::array<std::span<const std::int32_t>, 2> cells,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps);

} // namespace sparsitybuild
} // namespace dolfinx::fem
