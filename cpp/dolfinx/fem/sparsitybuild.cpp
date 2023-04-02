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
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  const DofMap& map0 = dofmaps[0].get();
  const DofMap& map1 = dofmaps[1].get();
  for (auto c : cells)
    pattern.insert(map0.cell_dofs(c), map1.cell_dofs(c));
}
//-----------------------------------------------------------------------------
void sparsitybuild::interior_facets(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps)
{
  const int D = topology.dim();
  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Topology facets have not been created.");

  auto connectivity = topology.connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Array to store macro-dofs
  std::array<std::vector<std::int32_t>, 2> macro_dofs;

  // Loop over owned facets
  auto map = topology.index_map(D - 1);
  assert(map);
  for (int f = 0; f < map->size_local(); ++f)
  {
    if (auto cells = connectivity->links(f); cells.size() == 2)
    {
      // Tabulate dofs for each dimension on macro element
      const int cell0 = cells[0];
      const int cell1 = cells[1];
      for (std::size_t i = 0; i < 2; i++)
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
