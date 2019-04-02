// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ElementDofLayout.h"
#include "ReferenceCellTopology.h"
#include <array>
#include <dolfin/mesh/CellType.h>
#include <map>
#include <set>
#include <ufc.h>

#include <iostream>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
ElementDofLayout::ElementDofLayout(
    int block_size, std::vector<std::vector<std::set<int>>> entity_dofs,
    std::vector<int> parent_map,
    std::vector<std::shared_ptr<ElementDofLayout>> sub_dofmaps,
    const mesh::CellType& cell_type)
    : _parent_map(parent_map), _block_size(block_size), _num_dofs(0),
      _entity_dofs(entity_dofs), _sub_dofmaps(sub_dofmaps)
{
  // TODO: Handle global support dofs

  // TODO: Add reference cell class to allow entity_closure_dofs to be
  //       removed as argument to constructor

  // dof = _entity_dofs[dim][entity_index][i]
  _num_entity_dofs.fill(0);
  for (std::size_t dim = 0; dim < entity_dofs.size(); ++dim)
  {
    assert(!entity_dofs[dim].empty());
    _num_entity_dofs[dim] = entity_dofs[dim][0].size();
    for (std::size_t entity_index = 0; entity_index < entity_dofs[dim].size();
         ++entity_index)
    {
      _num_dofs += entity_dofs[dim][entity_index].size();
    }
  }

  // dof = _entity_dofs[dim][entity_index][i]

  dolfin::CellType _cell = dolfin::CellType::point;
  if (cell_type.cell_type() == mesh::CellType::Type::interval)
    _cell = dolfin::CellType::interval;
  else if (cell_type.cell_type() == mesh::CellType::Type::triangle)
    _cell = dolfin::CellType::triangle;
  else if (cell_type.cell_type() == mesh::CellType::Type::quadrilateral)
    _cell = dolfin::CellType::quadrilateral;
  else if (cell_type.cell_type() == mesh::CellType::Type::tetrahedron)
    _cell = dolfin::CellType::tetrahedron;
  else if (cell_type.cell_type() == mesh::CellType::Type::hexahedron)
    _cell = dolfin::CellType::hexahedron;
  else
    throw std::runtime_error("Ooops");

  const int* num_entities = ReferenceCellTopology::num_entities(_cell);
  assert(num_entities);
  const ReferenceCellTopology::Edge* edge_v
      = ReferenceCellTopology::get_edge_vertices(_cell);
  const ReferenceCellTopology::Face* face_e
      = ReferenceCellTopology::get_face_edges(_cell);

  // Compute closure entities
  // [dim, entity] -> closure{(sub_dim, sub_entity)}
  std::map<std::array<int, 2>, std::map<int, std::set<int>>> entity_closure;
  for (int dim = 0; dim < (int)entity_dofs.size(); ++dim)
  {
    for (int entity = 0; entity < (int)entity_dofs[dim].size(); ++entity)
    {
      // Add self
      entity_closure[{dim, entity}][dim].insert(entity);

      if (dim == 3)
      {
        // Add all sub-entities
        for (int f = 0; f < num_entities[2]; ++f)
          entity_closure[{dim, entity}][2].insert(f);
        for (int e = 0; e < num_entities[1]; ++e)
          entity_closure[{dim, entity}][1].insert(e);
        for (int v = 0; v < num_entities[0]; ++v)
          entity_closure[{dim, entity}][0].insert(v);
      }

      if (dim == 2)
      {
        assert(face_e);
        CellType face_type = ReferenceCellTopology::entity_type(_cell, 2);
        const int num_edges = ReferenceCellTopology::num_edges(face_type);
        for (int e = 0; e < num_edges; ++e)
        {
          // Add edge
          const int edge_index = face_e[entity][e];
          entity_closure[{dim, entity}][1].insert(edge_index);
          for (int v = 0; v < 2; ++v)
          {
            // Add vertex connected to edge
            entity_closure[{dim, entity}][0].insert(edge_v[edge_index][v]);
          }
        }
      }

      if (dim == 1)
      {
        entity_closure[{dim, entity}][0].insert(edge_v[entity][0]);
        entity_closure[{dim, entity}][0].insert(edge_v[entity][1]);
      }
    }
  }

  // dof = _entity_dofs[dim][entity_index][i]
  _entity_closure_dofs = entity_dofs;
  for (auto entity : entity_closure)
  {
    const int dim = entity.first[0];
    const int index = entity.first[1];
    for (auto sub_entity : entity.second)
    {
      const int subdim = sub_entity.first;
      for (auto sub_index : sub_entity.second)
      {
        _entity_closure_dofs[dim][index].insert(
            entity_dofs[subdim][sub_index].begin(),
            entity_dofs[subdim][sub_index].end());
      }
    }
  }

  _num_entity_closure_dofs.fill(0);
  for (std::size_t dim = 0; dim < _entity_closure_dofs.size(); ++dim)
  {
    assert(!_entity_closure_dofs[dim].empty());
    _num_entity_closure_dofs[dim] = _entity_closure_dofs[dim][0].size();
  }
}
//-----------------------------------------------------------------------------
int ElementDofLayout::num_dofs() const { return _num_dofs; }
//-----------------------------------------------------------------------------
int ElementDofLayout::num_entity_dofs(unsigned int dim) const
{
  assert(dim < _num_entity_dofs.size());
  return _num_entity_dofs[dim];
}
//-----------------------------------------------------------------------------
int ElementDofLayout::num_entity_closure_dofs(unsigned int dim) const
{
  assert(dim < _num_entity_closure_dofs.size());
  return _num_entity_closure_dofs[dim];
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::set<int>>>&
ElementDofLayout::entity_dofs() const
{
  return _entity_dofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::set<int>>>&
ElementDofLayout::entity_closure_dofs() const
{
  return _entity_closure_dofs;
}
//-----------------------------------------------------------------------------
int ElementDofLayout::num_sub_dofmaps() const { return _sub_dofmaps.size(); }
//-----------------------------------------------------------------------------
std::shared_ptr<const ElementDofLayout>
ElementDofLayout::sub_dofmap(const std::vector<std::size_t>& component) const
{
  if (component.size() == 0)
    throw std::runtime_error("No sub dofmap specified");
  if (component[0] >= _sub_dofmaps.size())
    throw std::runtime_error("Invalid sub dofmap specified");

  std::shared_ptr<const ElementDofLayout> current = _sub_dofmaps[component[0]];
  for (unsigned int i = 1; i < component.size(); ++i)
  {
    const int idx = component[i];
    if (idx >= (int)current->_sub_dofmaps.size())
      throw std::runtime_error("Invalid component");
    current = _sub_dofmaps[idx];
  }
  return current;
}
//-----------------------------------------------------------------------------
std::vector<int> ElementDofLayout::sub_dofmap_mapping(
    const std::vector<std::size_t>& component) const
{
  // Fill up a list of parent dofs, from which subdofmap will select
  std::vector<int> dof_list(_num_dofs);
  std::iota(dof_list.begin(), dof_list.end(), 0);

  const ElementDofLayout* element_dofmap_current = this;
  for (auto i : component)
  {
    // Switch to sub-dofmap
    assert(element_dofmap_current);
    if (i >= element_dofmap_current->_sub_dofmaps.size())
      throw std::runtime_error("Invalid component");
    element_dofmap_current = _sub_dofmaps[i].get();

    std::vector<int> dof_list_new(element_dofmap_current->_num_dofs);
    for (unsigned int j = 0; j < dof_list_new.size(); ++j)
      dof_list_new[j] = dof_list[element_dofmap_current->_parent_map[j]];
    dof_list = dof_list_new;
  }

  return dof_list;
}
//-----------------------------------------------------------------------------
int ElementDofLayout::block_size() const { return _block_size; }
//-----------------------------------------------------------------------------
bool ElementDofLayout::is_view() const { return !_parent_map.empty(); }
//-----------------------------------------------------------------------------
