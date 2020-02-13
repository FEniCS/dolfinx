// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPatternBuilder.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::cells(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const fem::DofMap*, 2> dofmaps)
{
  assert(dofmaps[0]);
  assert(dofmaps[1]);
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  for (int c = 0; c < cells->num_nodes(); ++c)
    pattern.insert(dofmaps[0]->cell_dofs(c), dofmaps[1]->cell_dofs(c));
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::interior_facets(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const fem::DofMap*, 2> dofmaps)
{
  assert(dofmaps[0]);
  assert(dofmaps[1]);

  const int D = topology.dim();
  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Topology facets have not been created.");

  auto connectivity = topology.connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Array to store macro-dofs, if required (for interior facets)
  std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2> macro_dofs;

  // Loop over owned facets
  auto map = topology.index_map(D - 1);
  assert(map);
  assert(map->block_size == 1);
  const std::int32_t num_facets = map->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    // Get cells incident with facet
    auto cells = connectivity->links(f);
    // Proceed to next facet if only ony connection
    if (cells.rows() == 1)
      continue;

    // Tabulate dofs for each dimension on macro element
    assert(cells.rows() == 2);
    const int cell0 = cells[0];
    const int cell1 = cells[1];
    for (std::size_t i = 0; i < 2; i++)
    {
      auto cell_dofs0 = dofmaps[i]->cell_dofs(cell0);
      auto cell_dofs1 = dofmaps[i]->cell_dofs(cell1);
      macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());
      std::copy(cell_dofs0.data(), cell_dofs0.data() + cell_dofs0.size(),
                macro_dofs[i].data());
      std::copy(cell_dofs1.data(), cell_dofs1.data() + cell_dofs1.size(),
                macro_dofs[i].data() + cell_dofs0.size());
    }

    pattern.insert(macro_dofs[0], macro_dofs[1]);
  }
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::exterior_facets(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const fem::DofMap*, 2> dofmaps)
{
  const int D = topology.dim();
  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Topology facets have not been created.");

  auto connectivity = topology.connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Loop over owned facets
  auto map = topology.index_map(D - 1);
  assert(map);
  assert(map->block_size == 1);
  const std::int32_t num_facets = map->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    // Proceed to next facet if we have an interior facet
    if (connectivity->num_links(f) == 2)
      continue;

    auto cells = connectivity->links(f);
    assert(cells.rows() == 1);
    pattern.insert(dofmaps[0]->cell_dofs(cells[0]),
                   dofmaps[1]->cell_dofs(cells[0]));
  }
}
//-----------------------------------------------------------------------------
