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
  // Copy total number of dofs from ufc
  _num_dofs = dofmap.num_element_support_dofs + dofmap.num_global_support_dofs;

  // Copy over number of dofs per entity type (and also closure dofs per entity
  // type)
  // FIXME: can we generate closure dofs automatically here (see below)?
  std::copy(dofmap.num_entity_dofs, dofmap.num_entity_dofs + 4,
            _num_entity_dofs);
  std::copy(dofmap.num_entity_closure_dofs, dofmap.num_entity_closure_dofs + 4,
            _num_entity_closure_dofs);

  // Fill entity dof indices
  const unsigned int cell_dim = cell_type.dim();
  _entity_dofs.resize(cell_dim + 1);
  _entity_closure_dofs.resize(cell_dim + 1);
  for (unsigned int dim = 0; dim < cell_dim + 1; ++dim)
  {
    unsigned int num_entities = cell_type.num_entities(dim);
    _entity_dofs[dim].resize(num_entities);
    _entity_closure_dofs[dim].resize(num_entities);
    for (unsigned int i = 0; i < num_entities; ++i)
    {
      _entity_dofs[dim][i].resize(_num_entity_dofs[dim]);
      dofmap.tabulate_entity_dofs(_entity_dofs[dim][i].data(), dim, i);

      _entity_closure_dofs[dim][i].resize(_num_entity_closure_dofs[dim]);
      dofmap.tabulate_entity_closure_dofs(_entity_closure_dofs[dim][i].data(),
                                          dim, i);
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

  // UFC dofmaps just use simple offset for each field
  // but this could be different for custom dofmaps
  int offset = 0;
  for (auto& sub_dm : sub_dofmaps)
  {
    sub_dm->_parent_map.resize(sub_dm->num_dofs());
    std::iota(sub_dm->_parent_map.begin(), sub_dm->_parent_map.end(), offset);
    offset += sub_dm->_parent_map.size();
  }

  // Check for "block structure".
  // This should ultimately be replaced, but keep for now to mimic existing
  // code
  _block_size = analyse_block_structure();
}
//-----------------------------------------------------------------------------
int ElementDofMap::analyse_block_structure() const
{
  // Must be at least two subdofmaps
  if (sub_dofmaps.size() < 2)
    return 1;

  for (const auto& dmi : sub_dofmaps)
  {
    // If any subdofmaps have subdofmaps themselves, ignore any potential block
    // structure
    if (dmi->sub_dofmaps.size() > 0)
      return 1;

    // Check number of dofs are the same for all subdofmaps
    for (std::size_t d = 0; d < 4; ++d)
    {
      if (sub_dofmaps[0]->_num_entity_dofs[d] != dmi->_num_entity_dofs[d])
        return 1;
    }
  }

  // All subdofmaps are simple, and have the same number of dofs.
  return sub_dofmaps.size();
}
//-----------------------------------------------------------------------------
const ElementDofMap&
ElementDofMap::sub_dofmap(const std::vector<std::size_t>& component) const
{
  const ElementDofMap* current(this);
  for (auto i : component)
  {
    if (i >= current->sub_dofmaps.size())
      throw std::runtime_error("Invalid component");
    current = &*sub_dofmaps[i];
  }
  return *current;
}
//-----------------------------------------------------------------------------
std::vector<int> ElementDofMap::sub_dofmap_mapping(
    const std::vector<std::size_t>& component) const
{
  // Fill up a list of parent dofs, from which subdofmap will select
  std::vector<int> doflist(_num_dofs);
  std::iota(doflist.begin(), doflist.end(), 0);

  const ElementDofMap* current(this);
  for (auto i : component)
  {
    // Switch to sub-dofmap
    if (i >= current->sub_dofmaps.size())
      throw std::runtime_error("Invalid component");
    current = &*sub_dofmaps[i];

    std::vector<int> new_doflist(current->_num_dofs);
    for (unsigned int j = 0; j < new_doflist.size(); ++j)
      new_doflist[j] = doflist[current->_parent_map[j]];
    doflist = new_doflist;
  }
  return doflist;
}
//-----------------------------------------------------------------------------
void ElementDofMap::calculate_closure_dofs(const mesh::CellType& cell_type)
{
  // FIXME: this calculates the number of dofs, but still need to
  // work out actual dofs

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
void ElementDofMap::get_cell_entity_map(const mesh::CellType& cell_type)
{
  // FIXME: this calculates some connectivity within a cell, needed
  // to work out closure dofs.

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
  // FIXME: delegate to celltype
  if (cell_type.dim() == 3)
  {
    // Create a triangle (if tetrahedron) or quad (if hex)
    std::unique_ptr<mesh::CellType> facet_type(
        mesh::CellType::create(cell_type.facet_type()));

    // Index of each edge on each facet
    boost::multi_array<std::int32_t, 2> facet_edges(
        boost::extents[cell_type.num_entities(2)][facet_type->num_entities(1)]);

    // Create the edges of the facet and compare to the edges of the cell
    boost::multi_array<std::int32_t, 2> facet_edge_vertices;
    for (unsigned int i = 0; i < entity_vertices[2].shape()[0]; ++i)
    {
      std::vector<std::int32_t> facet_vertices(entity_vertices[2][i].begin(),
                                               entity_vertices[2][i].end());

      // Create all edges for this facet
      facet_type->create_entities(facet_edge_vertices, 1,
                                  facet_vertices.data());
      unsigned int j = 0;
      for (const auto& p : facet_edge_vertices)
      {
        // Find same edges in cell
        auto it = std::find(entity_vertices[1].begin(),
                            entity_vertices[1].end(), p);
        assert(it != entity_vertices[1].end());
        int idx = it - entity_vertices[1].begin();
        facet_edges[i][j] = idx;
        ++j;
      }
    }
  }
}
//-----------------------------------------------------------------------------
