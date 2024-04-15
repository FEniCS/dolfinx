// Copyright (C) 2007-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "sparsitybuild.h"
#include "DofMap.h"
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

  // Iterate over facets
  std::vector<std::int32_t> macro_dofs0, macro_dofs1;
  for (std::size_t f = 0; f < cells[0].size(); f += 2)
  {
    // Test function dofs (sparsity pattern rows)
    auto dofs00 = dofmap0.cell_dofs(cells0[f]);
    auto dofs01 = dofmap0.cell_dofs(cells0[f + 1]);
    macro_dofs0.resize(dofs00.size() + dofs01.size());
    std::copy(dofs00.begin(), dofs00.end(), macro_dofs0.begin());
    std::copy(dofs01.begin(), dofs01.end(),
              std::next(macro_dofs0.begin(), dofs00.size()));

    // Trial function dofs (sparsity pattern columns)
    auto dofs10 = dofmap1.cell_dofs(cells1[f]);
    auto dofs11 = dofmap1.cell_dofs(cells1[f + 1]);
    macro_dofs1.resize(dofs10.size() + dofs11.size());
    std::copy(dofs10.begin(), dofs10.end(), macro_dofs1.begin());
    std::copy(dofs11.begin(), dofs11.end(),
              std::next(macro_dofs1.begin(), dofs10.size()));

    pattern.insert(macro_dofs0, macro_dofs1);
  }
}
//-----------------------------------------------------------------------------
