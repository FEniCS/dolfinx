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
    pattern.insert_local(dofmaps[0]->cell_dofs(c), dofmaps[1]->cell_dofs(c));
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
  for (int f = 0; f < connectivity->num_nodes(); ++f)
  {
    // Continue if facet is exterior facet
    if (topology.size_global({D - 1, D}, f) == 1)
      continue;

    // FIXME: sort out ghosting

    // Get cells incident with facet
    auto cells = connectivity->links(f);
    assert(cells.rows() == 2);
    const int cell0 = cells[0];
    const int cell1 = cells[1];

    // Tabulate dofs for each dimension on macro element
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

    pattern.insert_local(macro_dofs[0], macro_dofs[1]);
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

  for (int f = 0; f < connectivity->num_nodes(); ++f)
  {
    // Skip interior facets
    if (topology.size_global({D - 1, D}, f) > 1)
      continue;

    // FIXME: sort out ghosting

    assert(connectivity->num_links(f) == 1);
    auto cells = connectivity->links(f);
    const int cell = cells[0];

    pattern.insert_local(dofmaps[0]->cell_dofs(cell),
                         dofmaps[1]->cell_dofs(cell));
  }
}
//-----------------------------------------------------------------------------
