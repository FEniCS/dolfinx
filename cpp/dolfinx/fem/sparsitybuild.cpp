// Copyright (C) 2007-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "sparsitybuild.h"
#include "DofMap.h"
#include <algorithm>
#include <dolfinx/la/SparsityPattern.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
void sparsitybuild::cells(
    la::SparsityPattern& pattern,
    std::array<std::span<const std::int32_t>, 2> cells,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  assert(cells[0].size() == cells[1].size());
  const DofMap& map0 = dofmaps[0].get();
  const DofMap& map1 = dofmaps[1].get();
  for (std::size_t i = 0; i < cells[0].size(); ++i)
    pattern.insert(map0.cell_dofs(cells[0][i]), map1.cell_dofs(cells[1][i]));
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

  const std::size_t dmap0_size = dofmap0.cell_dofs(0).size();
  const std::size_t dmap1_size = dofmap1.cell_dofs(0).size();

  // Iterate over facets
  std::vector<std::int32_t> macro_dofs0, macro_dofs1;
  for (std::size_t f = 0; f < cells[0].size(); f += 2)
  {
    // Test function dofs (sparsity pattern rows)
    std::span<const std::int32_t> dofs00;
    std::span<const std::int32_t> dofs01;

    // When integrating over interfaces between two domains, the test function
    // might only be defined on one side, so we check which cells exist in the
    // test function domain
    if (cells0[f] >= 0)
      dofs00 = dofmap0.cell_dofs(cells0[f]);
    if (cells0[f + 1] >= 0)
      dofs01 = dofmap0.cell_dofs(cells0[f + 1]);
    macro_dofs0.resize(2 * dmap0_size);
    std::ranges::copy(dofs00, macro_dofs0.begin());
    std::ranges::copy(dofs01, std::next(macro_dofs0.begin(), dmap0_size));

    // Trial function dofs (sparsity pattern columns)
    std::span<const std::int32_t> dofs10;
    std::span<const std::int32_t> dofs11;

    // Check which cells exist in the trial function domain
    if (cells1[f] >= 0)
      dofs10 = dofmap1.cell_dofs(cells1[f]);
    if (cells1[f + 1] >= 0)
      dofs11 = dofmap1.cell_dofs(cells1[f + 1]);
    macro_dofs1.resize(2 * dmap1_size);
    std::ranges::copy(dofs10, macro_dofs1.begin());
    std::ranges::copy(dofs11, std::next(macro_dofs1.begin(), dmap1_size));

    pattern.insert(macro_dofs0, macro_dofs1);
  }
}
//-----------------------------------------------------------------------------
