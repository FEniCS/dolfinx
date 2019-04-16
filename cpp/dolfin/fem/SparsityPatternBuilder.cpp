// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPatternBuilder.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::cells(
    la::SparsityPattern& pattern, const mesh::Mesh& mesh,
    const std::array<const fem::GenericDofMap*, 2> dofmaps)
{
  assert(dofmaps[0]);
  assert(dofmaps[1]);
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    pattern.insert_local(dofmaps[0]->cell_dofs(cell.index()),
                         dofmaps[1]->cell_dofs(cell.index()));
  }
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::interior_facets(
    la::SparsityPattern& pattern, const mesh::Mesh& mesh,
    const std::array<const fem::GenericDofMap*, 2> dofmaps)
{
  assert(dofmaps[0]);
  assert(dofmaps[1]);

  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  // Array to store macro-dofs, if required (for interior facets)
  std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2> macro_dofs;

  for (auto& facet : mesh::MeshRange<mesh::Facet>(mesh))
  {
    // Continue if facet is exterior facet
    if (facet.num_global_entities(D) == 1)
      continue;

    // FIXME: sort out ghosting

    // Get cells incident with facet
    assert(facet.num_entities(D) == 2);
    const mesh::Cell cell0(mesh, facet.entities(D)[0]);
    const mesh::Cell cell1(mesh, facet.entities(D)[1]);

    // Tabulate dofs for each dimension on macro element
    for (std::size_t i = 0; i < 2; i++)
    {
      const auto cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
      const auto cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());
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
    la::SparsityPattern& pattern, const mesh::Mesh& mesh,
    const std::array<const fem::GenericDofMap*, 2> dofmaps)
{
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  for (auto& facet : mesh::MeshRange<mesh::Facet>(mesh))
  {
    // Skip interior facets
    if (facet.num_global_entities(D) > 1)
      continue;

    // FIXME: sort out ghosting

    assert(facet.num_entities(D) == 1);
    mesh::Cell cell(mesh, facet.entities(D)[0]);
    pattern.insert_local(dofmaps[0]->cell_dofs(cell.index()),
                         dofmaps[1]->cell_dofs(cell.index()));
  }
}
//-----------------------------------------------------------------------------
