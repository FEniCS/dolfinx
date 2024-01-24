// Copyright (C) 2007-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "sparsitybuild.h"
#include "DofMap.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
void sparsitybuild::cells(
    la::SparsityPattern& pattern, std::span<const std::int32_t> cells,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps,
    const std::array<std::function<std::int32_t(std::int32_t)>, 2>& cell_maps)
{
  const DofMap& map0 = dofmaps[0].get();
  const DofMap& map1 = dofmaps[1].get();
  for (auto c : cells)
    pattern.insert(map0.cell_dofs(cell_maps[0](c)),
                   map1.cell_dofs(cell_maps[1](c)));
}
//-----------------------------------------------------------------------------
void sparsitybuild::interior_facets(
    la::SparsityPattern& pattern, std::span<const std::int32_t> facets,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  std::array<std::vector<std::int32_t>, 2> macro_dofs;
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    int cell_0 = facets[index];
    int cell_1 = facets[index + 1];
    for (std::size_t i = 0; i < 2; ++i)
    {
      auto cell_dofs_0 = dofmaps[i].get().cell_dofs(cell_0);
      auto cell_dofs_1 = dofmaps[i].get().cell_dofs(cell_1);
      macro_dofs[i].resize(cell_dofs_0.size() + cell_dofs_1.size());
      std::copy(cell_dofs_0.begin(), cell_dofs_0.end(), macro_dofs[i].begin());
      std::copy(cell_dofs_1.begin(), cell_dofs_1.end(),
                std::next(macro_dofs[i].begin(), cell_dofs_0.size()));
    }
    pattern.insert(macro_dofs[0], macro_dofs[1]);
  }
}
//-----------------------------------------------------------------------------
