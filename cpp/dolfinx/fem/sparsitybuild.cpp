// Copyright (C) 2007-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "sparsitybuild.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/SparsityPattern.h>
#include <stdexcept>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
std::vector<std::pair<std::span<const int>, std::span<const int>>>
sparsitybuild::entity_closure_blocks(const ElementDofLayout& layout_rows,
                                     const ElementDofLayout& layout_cols)
{
  const std::vector<std::vector<std::vector<int>>>& edofs1
      = layout_rows.entity_dofs_all();
  const std::vector<std::vector<std::vector<int>>>& cdofs0
      = layout_cols.entity_closure_dofs_all();
  auto size = [](const std::vector<std::vector<int>>& e) { return e.size(); };
  if (!std::ranges::equal(edofs1, cdofs0, {}, size, size))
    throw std::invalid_argument("Dof layouts have different reference cells.");

  std::vector<std::pair<std::span<const int>, std::span<const int>>> blocks;
  for (std::size_t d = 0; d < edofs1.size(); ++d)
    for (std::size_t e = 0; e < edofs1[d].size(); ++e)
      if (!edofs1[d][e].empty())
        blocks.emplace_back(edofs1[d][e], cdofs0[d][e]);

  return blocks;
}
//-----------------------------------------------------------------------------
void sparsitybuild::entity_closure(
    la::SparsityPattern& pattern, std::span<const std::int32_t> cells,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  // As elsewhere in sparsitybuild, dofmaps[0] gives the rows
  const DofMap& map0 = dofmaps[0];
  const DofMap& map1 = dofmaps[1];
  const std::vector<std::pair<std::span<const int>, std::span<const int>>>
      blocks = entity_closure_blocks(map0.element_dof_layout(),
                                     map1.element_dof_layout());
  if (blocks.empty())
    return;

  // An entity is shared by several cells, and each visit would insert
  // the same block, so insert each one once. A block's rows are the
  // degrees-of-freedom of one entity, so its first row identifies it.
  std::vector<std::int8_t> seen(
      map0.index_map->size_local() + map0.index_map->num_ghosts(), 0);

  // Count what will be inserted, so that the pattern's cache is sized
  // once. Only the first row of each block is needed to count.
  std::size_t nblocks = 0, nrows = 0, ncols = 0;
  for (std::int32_t c : cells)
  {
    std::span<const std::int32_t> cell_rows = map0.cell_dofs(c);
    for (const auto& [rdofs, cdofs] : blocks)
    {
      if (std::int32_t r = cell_rows[rdofs[0]]; !seen[r])
      {
        seen[r] = 1;
        ++nblocks;
        nrows += rdofs.size();
        ncols += cdofs.size();
      }
    }
  }
  pattern.reserve_blocks(nblocks, nrows, ncols);

  std::ranges::fill(seen, 0);
  std::vector<std::int32_t> rows(map0.element_dof_layout().num_dofs()),
      cols(map1.element_dof_layout().num_dofs());
  for (std::int32_t c : cells)
  {
    std::span<const std::int32_t> cell_cols = map1.cell_dofs(c);
    std::span<const std::int32_t> cell_rows = map0.cell_dofs(c);
    for (const auto& [rdofs, cdofs] : blocks)
    {
      std::int32_t r = cell_rows[rdofs[0]];
      if (seen[r])
        continue;
      seen[r] = 1;
      std::ranges::transform(rdofs, rows.begin(),
                             [cell_rows](int d) { return cell_rows[d]; });
      std::ranges::transform(cdofs, cols.begin(),
                             [cell_cols](int d) { return cell_cols[d]; });
      pattern.insert(std::span(rows).first(rdofs.size()),
                     std::span(cols).first(cdofs.size()));
    }
  }
}
//-----------------------------------------------------------------------------
void sparsitybuild::interior_facets(
    la::SparsityPattern& pattern,
    std::array<std::span<const std::int32_t>, 2> cells,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  std::span<const std::int32_t> cells0 = cells[0];
  std::span<const std::int32_t> cells1 = cells[1];
  assert(cells0.size() == cells1.size());
  const DofMap& dofmap0 = dofmaps[0];
  const DofMap& dofmap1 = dofmaps[1];

  // Iterate over facets
  for (std::size_t f = 0; f < cells0.size(); f += 2)
  {
    // Test function dofs (sparsity pattern rows). A cell may not
    // exist on this side (e.g. an interface between two domains).
    std::span<const std::int32_t> dofs00
        = cells0[f] >= 0 ? dofmap0.cell_dofs(cells0[f])
                         : std::span<const std::int32_t>();
    std::span<const std::int32_t> dofs01
        = cells0[f + 1] >= 0 ? dofmap0.cell_dofs(cells0[f + 1])
                             : std::span<const std::int32_t>();

    // Trial function dofs (sparsity pattern columns)
    std::span<const std::int32_t> dofs10
        = cells1[f] >= 0 ? dofmap1.cell_dofs(cells1[f])
                         : std::span<const std::int32_t>();
    std::span<const std::int32_t> dofs11
        = cells1[f + 1] >= 0 ? dofmap1.cell_dofs(cells1[f + 1])
                             : std::span<const std::int32_t>();

    // Insert the four (test, trial) blocks directly, rather than via
    // a temporary buffer that could leak a previous facet's dofs
    pattern.insert(dofs00, dofs10);
    pattern.insert(dofs00, dofs11);
    pattern.insert(dofs01, dofs10);
    pattern.insert(dofs01, dofs11);
  }
}
//-----------------------------------------------------------------------------
