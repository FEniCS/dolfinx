// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPatternBuilder.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::cells(
    la::SparsityPattern& pattern, const mesh::Mesh& mesh,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs0,
    int dim0,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs1,
    int dim1)
{
  const int D = mesh.topology().dim();
  for (auto& cell : mesh::MeshRange(mesh, D))
  {
    const int index = cell.index();
    pattern.insert_local(dofs0.segment(dim0 * index, dim0),
                         dofs1.segment(dim1 * index, dim1));
  }
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::interior_facets(
    la::SparsityPattern& pattern, const mesh::Mesh& mesh,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs0,
    int dim0,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs1,
    int dim1)
{
  const std::size_t D = mesh.topology().dim();
  mesh.create_entities(D - 1);
  mesh.create_connectivity(D - 1, D);

  // Array to store macro-dofs, if required (for interior facets)
  std::shared_ptr<const mesh::Connectivity> connectivity
      = mesh.topology().connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2> macro_dofs;
  for (auto& facet : mesh::MeshRange(mesh, D - 1))
  {
    // Continue if facet is exterior facet
    if (connectivity->size_global(facet.index()) == 1)
      continue;

    // FIXME: sort out ghosting

    // Get cells incident with facet
    assert(connectivity->size(facet.index()) == 2);
    const mesh::MeshEntity cell0(mesh, D, facet.entities(D)[0]);
    const mesh::MeshEntity cell1(mesh, D, facet.entities(D)[1]);

    const int index_c0 = cell0.index();
    const int index_c1 = cell1.index();

    // Get dofmaps 0 for cells 0 and 1
    auto cell_dofs0_c0 = dofs0.segment(dim0 * index_c0, dim0);
    auto cell_dofs0_c1 = dofs0.segment(dim0 * index_c1, dim0);

    // Get dofmaps 1 for cells 0 and 1
    auto cell_dofs1_c0 = dofs1.segment(dim1 * index_c0, dim1);
    auto cell_dofs1_c1 = dofs1.segment(dim1 * index_c1, dim1);

    // Stack dofmaps for each cells
    macro_dofs[0].resize(cell_dofs0_c0.size() + cell_dofs0_c1.size());
    macro_dofs[1].resize(cell_dofs1_c0.size() + cell_dofs1_c1.size());

    macro_dofs[0].head(cell_dofs0_c0.size()) = cell_dofs0_c0;
    macro_dofs[0].tail(cell_dofs0_c1.size()) = cell_dofs0_c1;

    macro_dofs[1].head(cell_dofs1_c0.size()) = cell_dofs1_c0;
    macro_dofs[1].tail(cell_dofs1_c1.size()) = cell_dofs1_c1;

    // Insert into sparsity pattern
    pattern.insert_local(macro_dofs[0], macro_dofs[1]);
  }
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::exterior_facets(
    la::SparsityPattern& pattern, const mesh::Mesh& mesh,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs0,
    int dim0,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs1,
    int dim1)
{
  const std::size_t D = mesh.topology().dim();
  mesh.create_entities(D - 1);
  mesh.create_connectivity(D - 1, D);

  std::shared_ptr<const mesh::Connectivity> connectivity
      = mesh.topology().connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");
  for (auto& facet : mesh::MeshRange(mesh, D - 1))
  {
    // Skip interior facets
    if (connectivity->size_global(facet.index()) > 1)
      continue;

    // FIXME: sort out ghosting

    assert(connectivity->size(facet.index()) == 1);
    mesh::MeshEntity cell(mesh, D, facet.entities(D)[0]);
    const int index = cell.index();
    pattern.insert_local(dofs0.segment(dim0 * index, dim0),
                         dofs1.segment(dim1 * index, dim1));
  }
}
//-----------------------------------------------------------------------------
