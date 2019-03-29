// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ElementDofLayout.h"
#include <dolfin/mesh/CellType.h>
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
ElementDofLayout::ElementDofLayout(
    int block_size, std::vector<std::vector<std::vector<int>>> entity_dofs,
    std::vector<std::vector<std::vector<int>>> entity_closure_dofs,
    std::vector<int> parent_map,
    std::vector<std::shared_ptr<ElementDofLayout>> sub_dofmaps)
    : _parent_map(parent_map), _block_size(block_size), _num_dofs(0),
      _entity_dofs(entity_dofs), _entity_closure_dofs(entity_closure_dofs),
      _sub_dofmaps(sub_dofmaps)
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

  _num_entity_closure_dofs.fill(0);
  for (std::size_t dim = 0; dim < entity_closure_dofs.size(); ++dim)
  {
    assert(!entity_closure_dofs[dim].empty());
    _num_entity_closure_dofs[dim] = entity_closure_dofs[dim][0].size();
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
const std::vector<std::vector<std::vector<int>>>&
ElementDofLayout::entity_dofs() const
{
  return _entity_dofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::vector<int>>>&
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
