// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ElementDofMap.h"
#include <cstdlib>
#include <dolfin/mesh/CellType.h>
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
// Try to figure out block size. FIXME - replace elsewhere
int analyse_block_structure(
    const std::vector<std::shared_ptr<ElementDofMap>> sub_dofmaps)
{
  // Must be at least two subdofmaps
  if (sub_dofmaps.size() < 2)
    return 1;

  for (auto dmap : sub_dofmaps)
  {
    assert(dmap);

    // If any subdofmaps have subdofmaps themselves, ignore any
    // potential block structure
    if (dmap->num_sub_dofmaps() > 0)
      return 1;

    // Check number of dofs are the same for all subdofmaps
    for (int d = 0; d < 4; ++d)
    {
      if (sub_dofmaps[0]->num_entity_dofs(d) != dmap->num_entity_dofs(d))
        return 1;
    }
  }

  // All subdofmaps are simple, and have the same number of dofs
  return sub_dofmaps.size();
}
} // namespace

//-----------------------------------------------------------------------------
ElementDofMap::ElementDofMap(const ufc_dofmap& dofmap,
                             const mesh::CellType& cell_type)
{
  // Get total number of dofs from ufc
  _num_dofs = dofmap.num_element_support_dofs + dofmap.num_global_support_dofs;

  // Copy over number of dofs per entity type (and also closure dofs per
  // entity type)
  // FIXME: can we generate closure dofs automatically here (see below)?
  std::copy(dofmap.num_entity_dofs, dofmap.num_entity_dofs + 4,
            _num_entity_dofs.data());
  std::copy(dofmap.num_entity_closure_dofs, dofmap.num_entity_closure_dofs + 4,
            _num_entity_closure_dofs.data());

  // Fill entity dof indices
  const int tdim = cell_type.dim();
  _entity_dofs.resize(tdim + 1);
  _entity_closure_dofs.resize(tdim + 1);
  for (int dim = 0; dim <= tdim; ++dim)
  {
    int num_entities = cell_type.num_entities(dim);
    _entity_dofs[dim].resize(num_entities);
    _entity_closure_dofs[dim].resize(num_entities);
    for (int i = 0; i < num_entities; ++i)
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
    _sub_dofmaps.push_back(
        std::make_shared<ElementDofMap>(*sub_dofmap, cell_type));
    std::free(sub_dofmap);
  }

  // UFC dofmaps just use simple offset for each field but this could be
  // different for custom dofmaps
  int offset = 0;
  for (auto& sub_dofmap : _sub_dofmaps)
  {
    sub_dofmap->_parent_map.resize(sub_dofmap->num_dofs());
    std::iota(sub_dofmap->_parent_map.begin(), sub_dofmap->_parent_map.end(),
              offset);
    offset += sub_dofmap->_parent_map.size();
  }

  // Check for "block structure". This should ultimately be replaced,
  // but keep for now to mimic existing code
  _block_size = analyse_block_structure(_sub_dofmaps);
}
//-----------------------------------------------------------------------------
int ElementDofMap::num_dofs() const { return _num_dofs; }
//-----------------------------------------------------------------------------
int ElementDofMap::num_entity_dofs(int dim) const
{
  assert(dim < 4);
  return _num_entity_dofs[dim];
}
//-----------------------------------------------------------------------------
int ElementDofMap::num_entity_closure_dofs(int dim) const
{
  assert(dim < 4);
  return _num_entity_closure_dofs[dim];
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::vector<int>>>&
ElementDofMap::entity_dofs() const
{
  return _entity_dofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::vector<int>>>&
ElementDofMap::entity_closure_dofs() const
{
  return _entity_closure_dofs;
}
//-----------------------------------------------------------------------------
int ElementDofMap::num_sub_dofmaps() const { return _sub_dofmaps.size(); }
//-----------------------------------------------------------------------------
std::shared_ptr<const ElementDofMap>
ElementDofMap::sub_dofmap(const std::vector<std::size_t>& component) const
{
  if (component.size() == 0)
    throw std::runtime_error("No sub dofmap specified");
  if (component[0] >= _sub_dofmaps.size())
    throw std::runtime_error("Invalid sub dofmap specified");

  std::shared_ptr<const ElementDofMap> current = _sub_dofmaps[component[0]];
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
std::vector<int> ElementDofMap::sub_dofmap_mapping(
    const std::vector<std::size_t>& component) const
{
  // Fill up a list of parent dofs, from which subdofmap will select
  std::vector<int> dof_list(_num_dofs);
  std::iota(dof_list.begin(), dof_list.end(), 0);

  const ElementDofMap* element_dofmap_current = this;
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
int ElementDofMap::block_size() const { return _block_size; }
//-----------------------------------------------------------------------------
