// Copyright (C) 2007-2019 Garth N. Wells
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
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps,
    const std::array<const std::function<std::int32_t(std::span<const int>)>,
                     2>& cell_maps)
{
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  for (int c = 0; c < cells->num_nodes(); ++c)
  {
    // NOTE Since this loops over all cells, not just the integration domains
    // of the integrals in the form, c_0 or c_1 may not exist. In this case,
    // nothing needs to be inserted into the sparsity pattern.
    // TODO See if it is necessary to add check that c_0 and c_1 are >= 0 before
    // inserting into pattern
    std::int32_t c_0 = cell_maps[0](std::array<std::int32_t, 1>{c});
    std::int32_t c_1 = cell_maps[1](std::array<std::int32_t, 1>{c});
    pattern.insert(dofmaps[0].get().cell_dofs(c_0),
                   dofmaps[1].get().cell_dofs(c_1));
  }
}
//-----------------------------------------------------------------------------
void sparsitybuild::cells(
    la::SparsityPattern& pattern, const std::span<const std::int32_t>& cells,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps,
    const std::array<const std::function<std::int32_t(std::span<const int>)>,
                     2>& cell_maps)
{
  for (std::int32_t c : cells)
  {
    std::int32_t c_0 = cell_maps[0](std::array<std::int32_t, 1>{c});
    std::int32_t c_1 = cell_maps[1](std::array<std::int32_t, 1>{c});
    pattern.insert(dofmaps[0].get().cell_dofs(c_0),
                   dofmaps[1].get().cell_dofs(c_1));
  }
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

  // Array to store macro-dofs, if required (for interior facets)
  std::array<std::vector<std::int32_t>, 2> macro_dofs;

  // Loop over owned facets
  auto map = topology.index_map(D - 1);
  assert(map);
  const std::int32_t num_facets = map->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    // Get cells incident with facet
    auto cells = connectivity->links(f);

    // Proceed to next facet if only connection
    if (cells.size() == 1)
      continue;

    // Tabulate dofs for each dimension on macro element
    assert(cells.size() == 2);
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
//-----------------------------------------------------------------------------
void sparsitybuild::interior_facets(
    la::SparsityPattern& pattern, const std::span<const std::int32_t>& facets,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps,
    const std::array<const std::function<std::int32_t(std::span<const int>)>,
                     2>& facet_maps)
{
  std::array<std::vector<std::int32_t>, 2> macro_dofs;
  for (std::size_t index = 0; index < facets.size(); index += 4)
  {
    for (std::size_t i = 0; i < 2; ++i)
    {
      // TODO Use span to simplify (see assemblers)
      std::span<const std::int32_t> facet_0 = facets.subspan(index, 2);
      std::span<const std::int32_t> facet_1 = facets.subspan(index + 2, 2);
      const std::int32_t cell_0 = facet_maps[i](facet_0);
      const std::int32_t cell_1 = facet_maps[i](facet_1);

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
void sparsitybuild::exterior_facets(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps,
    const std::array<const std::function<std::int32_t(std::span<const int>)>,
                     2>& facet_maps)
{
  const int D = topology.dim();
  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Topology facets have not been created.");

  auto c_to_f = topology.connectivity(D, D - 1);
  auto f_to_c = topology.connectivity(D - 1, D);
  if (!c_to_f or !f_to_c)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Loop over owned facets
  auto map = topology.index_map(D - 1);
  assert(map);
  const std::int32_t num_facets = map->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    // Proceed to next facet if we have an interior facet
    if (f_to_c->num_links(f) == 2)
      continue;

    auto cells = f_to_c->links(f);
    assert(cells.size() == 1);

    std::int32_t cell = cells[0];
    // Get the local facet index
    auto cell_facets = c_to_f->links(cell);
    auto facet_it = std::find(cell_facets.begin(), cell_facets.end(), f);
    assert(facet_it != cell_facets.end());
    int local_f = std::distance(cell_facets.begin(), facet_it);

    pattern.insert(dofmaps[0].get().cell_dofs(facet_maps[0](
                       std::array<std::int32_t, 2>{cell, local_f})),
                   dofmaps[1].get().cell_dofs(facet_maps[1](
                       std::array<std::int32_t, 2>{cell, local_f})));
  }
}
//-----------------------------------------------------------------------------
void sparsitybuild::exterior_facets(
    la::SparsityPattern& pattern, const std::span<const std::int32_t>& facets,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps,
    const std::array<const std::function<std::int32_t(std::span<const int>)>,
                     2>& facet_maps)
{
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    std::span<const std::int32_t> facet = facets.subspan(index, 2);
    pattern.insert(dofmaps[0].get().cell_dofs(facet_maps[0](facet)),
                   dofmaps[1].get().cell_dofs(facet_maps[1](facet)));
  }
}
//-----------------------------------------------------------------------------
