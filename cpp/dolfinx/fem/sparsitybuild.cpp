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
    la::SparsityPattern& pattern, std::span<const std::int32_t> facets,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  std::array<std::vector<std::int32_t>, 2> macro_dofs;
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    std::int32_t cell0 = facets[index];
    std::int32_t cell1 = facets[index + 1];
    for (std::size_t i = 0; i < 2; ++i)
    {
      auto cell_dofs0 = dofmaps[i].get().cell_dofs(cell0);
      auto cell_dofs1 = dofmaps[i].get().cell_dofs(cell1);
      macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());
      std::copy(cell_dofs0.begin(), cell_dofs0.end(), macro_dofs[i].begin());
      std::copy(cell_dofs1.begin(), cell_dofs1.end(),
                std::next(macro_dofs[i].begin(), cell_dofs0.size()));
    }
    pattern.insert(macro_dofs[0], macro_dofs[1]);
  }
}
//-----------------------------------------------------------------------------
