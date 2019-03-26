// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ElementDofMap.h"
#include <cstdlib>
#include <dolfin/mesh/CellType.h>
#include <iostream>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
ElementDofMap::ElementDofMap(const ufc_dofmap& dofmap,
                             const mesh::CellType& cell_type)
{
  _num_dofs = dofmap.num_element_support_dofs + dofmap.num_global_support_dofs;

  // Copy over number of dofs per entity
  std::copy(dofmap.num_entity_dofs, dofmap.num_entity_dofs + 4,
            _num_entity_dofs);

  // Fill entity dof indices
  const unsigned int cell_dim = cell_type.dim();
  _entity_dofs.resize(cell_dim + 1);
  for (unsigned int dim = 0; dim < cell_dim + 1; ++dim)
  {
    unsigned int num_entities = cell_type.num_entities(dim);
    _entity_dofs[dim].resize(num_entities);
    for (unsigned int i = 0; i < num_entities; ++i)
    {
      _entity_dofs[dim][i].resize(_num_entity_dofs[dim]);
      dofmap.tabulate_entity_dofs(_entity_dofs[dim][i].data(), dim, i);
    }
  }

  // Fill all subdofmaps
  for (int i = 0; i < dofmap.num_sub_dofmaps; ++i)
  {
    ufc_dofmap* sub_dofmap = dofmap.create_sub_dofmap(i);
    sub_dofmaps.push_back(
        std::make_unique<ElementDofMap>(*sub_dofmap, cell_type));
    std::free(sub_dofmap);
  }

  // Closure dofs
  calculate_closure_dofs(cell_type);

  // Check for "block structure".
  // This should ultimately be replaced, but keep
  // for now to mimic existing code
  _block_size = analyse_block_structure();
}
//-----------------------------------------------------------------------------
int ElementDofMap::analyse_block_structure() const
{
  if (sub_dofmaps.size() < 2)
    return 1;

  for (const auto& dmi : sub_dofmaps)
  {
    if (dmi->sub_dofmaps.size() > 0)
      return 1;

    for (std::size_t d = 0; d < 4; ++d)
    {
      if (sub_dofmaps[0]->_num_entity_dofs[d] != dmi->_num_entity_dofs[d])
        return 1;
    }
  }

  return sub_dofmaps.size();
}
//-----------------------------------------------------------------------------
std::vector<int> ElementDofMap::tabulate_entity_dofs(unsigned int dim,
                                                     unsigned int i) const
{
  assert(dim < _entity_dofs.size());
  assert(i < _entity_dofs[dim].size());

  return _entity_dofs[dim][i];
}
//-----------------------------------------------------------------------------
void ElementDofMap::calculate_closure_dofs(const mesh::CellType& cell_type)
{

  // Copy entity dofs, and add to them
  std::vector<std::vector<std::vector<int>>> _entity_closure_dofs
      = _entity_dofs;

  std::vector<int> num_closure_dofs(_entity_dofs.size());

  for (unsigned int dim = 0; dim < num_closure_dofs.size(); ++dim)
  {
    std::unique_ptr<mesh::CellType> entity_cell_type(
        mesh::CellType::create(cell_type.entity_type(dim)));

    num_closure_dofs[dim] = _num_entity_dofs[dim];
    for (unsigned int j = 0; j < dim; ++j)
    {
      const unsigned int num_entities_j = entity_cell_type->num_entities(j);
      num_closure_dofs[dim] += num_entities_j * _num_entity_dofs[j];
    }
    std::cout << "closure [" << dim << "] = " << num_closure_dofs[dim]
              << std::endl;
  }
}
//-----------------------------------------------------------------------------
void get_cell_entity_map(const mesh::CellType& cell_type)
{
  const int nv = cell_type.num_vertices();

  std::vector<boost::multi_array<std::int32_t, 2>> entity_vertices(
      cell_type.dim());

  // Create list of vertices
  entity_vertices[0].resize(boost::extents[nv][1]);
  std::iota(entity_vertices[0].data(), entity_vertices[0].data() + nv, 0);

  // Get entity->vertex mapping
  for (unsigned int dim = 1; dim < cell_type.dim(); ++dim)
  {
    std::cout << "Creating entities of dim = " << dim << "\n";
    cell_type.create_entities(entity_vertices[dim], dim,
                              entity_vertices[0].data());
  }

  // Work out the face->edge relation in 3D.
  boost::multi_array<std::int32_t, 2> face_edges;
  if (cell_type.dim() == 3)
  {
    std::unique_ptr<mesh::CellType> facet_type(
        mesh::CellType::create(cell_type.facet_type()));

    face_edges.resize(
        boost::extents[cell_type.num_entities(2)][facet_type->num_entities(1)]);

    for (unsigned int i = 0; i < entity_vertices[2].shape()[0]; ++i)
    {
      facet_type->create_entities(edge_vertices, 1,
                                  entity_vertices[2][i].data());
    }
  }
}
