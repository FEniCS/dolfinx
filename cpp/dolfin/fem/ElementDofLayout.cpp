// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ElementDofLayout.h"
#include <array>
#include <map>
#include <numeric>
#include <set>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
ElementDofLayout::ElementDofLayout(
    int block_size, const std::vector<std::vector<std::set<int>>>& entity_dofs,
    const std::vector<int>& parent_map,
    const std::vector<std::shared_ptr<const ElementDofLayout>> sub_dofmaps,
    const mesh::CellType cell_type)
    : _block_size(block_size), _parent_map(parent_map), _num_dofs(0),
      _entity_dofs(entity_dofs), _sub_dofmaps(sub_dofmaps)
{
  // TODO: Handle global support dofs

  // Compute closure entities
  // [dim, entity] -> closure{sub_dim, (sub_entities)}
  std::map<std::array<int, 2>, std::vector<std::set<int>>> entity_closure
      = mesh::cell_entity_closure(cell_type);

  // dof = _entity_dofs[dim][entity_index][i]
  _entity_closure_dofs = entity_dofs;
  for (auto entity : entity_closure)
  {
    const int dim = entity.first[0];
    const int index = entity.first[1];
    int subdim = 0;
    for (auto sub_entity : entity.second)
    {
      for (auto sub_index : sub_entity)
      {
        _entity_closure_dofs[dim][index].insert(
            entity_dofs[subdim][sub_index].begin(),
            entity_dofs[subdim][sub_index].end());
      }
      ++subdim;
    }
  }

  // dof = _entity_dofs[dim][entity_index][i]
  _num_entity_dofs.fill(0);
  _num_entity_closure_dofs.fill(0);
  assert(entity_dofs.size() == _entity_closure_dofs.size());
  for (std::size_t dim = 0; dim < entity_dofs.size(); ++dim)
  {
    assert(!entity_dofs[dim].empty());
    assert(!_entity_closure_dofs[dim].empty());
    _num_entity_dofs[dim] = entity_dofs[dim][0].size();
    _num_entity_closure_dofs[dim] = _entity_closure_dofs[dim][0].size();
    for (std::size_t entity_index = 0; entity_index < entity_dofs[dim].size();
         ++entity_index)
    {
      _num_dofs += entity_dofs[dim][entity_index].size();
    }
  }
}
//-----------------------------------------------------------------------------
ElementDofLayout::ElementDofLayout(const ElementDofLayout& element_dof_layout,
                                   bool reset_parent)
    : ElementDofLayout(element_dof_layout)
{
  _parent_map.clear();
}
//-----------------------------------------------------------------------------
int ElementDofLayout::num_dofs() const { return _num_dofs; }
//-----------------------------------------------------------------------------
int ElementDofLayout::num_entity_dofs(int dim) const
{
  return _num_entity_dofs.at(dim);
}
//-----------------------------------------------------------------------------
int ElementDofLayout::num_entity_closure_dofs(int dim) const
{
  return _num_entity_closure_dofs.at(dim);
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, 1>
ElementDofLayout::entity_dofs(int entity_dim, int cell_entity_index) const
{
  const std::set<int>& edofs
      = _entity_dofs.at(entity_dim).at(cell_entity_index);
  Eigen::Array<int, Eigen::Dynamic, 1> dofs(edofs.size());
  std::copy(edofs.begin(), edofs.end(), dofs.data());
  return dofs;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, 1>
ElementDofLayout::entity_closure_dofs(int entity_dim,
                                      int cell_entity_index) const
{
  const std::set<int>& edofs
      = _entity_closure_dofs.at(entity_dim).at(cell_entity_index);
  Eigen::Array<int, Eigen::Dynamic, 1> dofs(edofs.size());
  std::copy(edofs.begin(), edofs.end(), dofs.data());
  return dofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::set<int>>>&
ElementDofLayout::entity_dofs_all() const
{
  return _entity_dofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::set<int>>>&
ElementDofLayout::entity_closure_dofs_all() const
{
  return _entity_closure_dofs;
}
//-----------------------------------------------------------------------------
int ElementDofLayout::num_sub_dofmaps() const { return _sub_dofmaps.size(); }
//-----------------------------------------------------------------------------
std::shared_ptr<const ElementDofLayout>
ElementDofLayout::sub_dofmap(const std::vector<int>& component) const
{
  if (component.size() == 0)
    throw std::runtime_error("No sub dofmap specified");
  std::shared_ptr<const ElementDofLayout> current
      = _sub_dofmaps.at(component[0]);
  for (std::size_t i = 1; i < component.size(); ++i)
  {
    const int idx = component[i];
    current = _sub_dofmaps.at(idx);
  }
  return current;
}
//-----------------------------------------------------------------------------
std::vector<int>
ElementDofLayout::sub_view(const std::vector<int>& component) const
{
  // Fill up a list of parent dofs, from which subdofmap will select
  std::vector<int> dof_list(_num_dofs);
  std::iota(dof_list.begin(), dof_list.end(), 0);

  const ElementDofLayout* element_dofmap_current = this;
  for (int i : component)
  {
    // Switch to sub-dofmap
    assert(element_dofmap_current);
    if (i >= (int)element_dofmap_current->_sub_dofmaps.size())
      throw std::runtime_error("Invalid component");
    element_dofmap_current = _sub_dofmaps.at(i).get();

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
