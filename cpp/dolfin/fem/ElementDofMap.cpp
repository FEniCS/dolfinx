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

namespace
{
// Try to figure out block size. FIXME - replace elsewhere
int analyse_block_structure(
    const std::vector<std::shared_ptr<ElementDofMap>> sub_dofmaps)
{
  // Must be at least two subdofmaps
  if (sub_dofmaps.size() < 2)
    return 1;

  for (const auto& dmi : sub_dofmaps)
  {
    // If any subdofmaps have subdofmaps themselves, ignore any
    // potential block structure
    if (dmi->num_sub_dofmaps() > 0)
      return 1;

    // Check number of dofs are the same for all subdofmaps
    for (int d = 0; d < 4; ++d)
    {
      if (sub_dofmaps[0]->num_entity_dofs(d) != dmi->num_entity_dofs(d))
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
  // Initialise an "ElementDofMap" from a ufc_dofmap

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
  const int cell_dim = cell_type.dim();
  _entity_dofs.resize(cell_dim + 1);
  _entity_closure_dofs.resize(cell_dim + 1);
  for (int dim = 0; dim < cell_dim + 1; ++dim)
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
  for (auto& sub_dm : _sub_dofmaps)
  {
    sub_dm->_parent_map.resize(sub_dm->num_dofs());
    std::iota(sub_dm->_parent_map.begin(), sub_dm->_parent_map.end(), offset);
    offset += sub_dm->_parent_map.size();
  }

  // Check for "block structure". This should ultimately be replaced,
  // but keep for now to mimic existing code
  _block_size = analyse_block_structure(_sub_dofmaps);
}
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
    if (idx >= current->_sub_dofmaps.size())
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
  std::vector<int> doflist(_num_dofs);
  std::iota(doflist.begin(), doflist.end(), 0);

  const ElementDofMap* current(this);
  for (auto i : component)
  {
    // Switch to sub-dofmap
    if (i >= current->_sub_dofmaps.size())
      throw std::runtime_error("Invalid component");
    current = _sub_dofmaps[i].get();

    std::vector<int> new_doflist(current->_num_dofs);
    for (unsigned int j = 0; j < new_doflist.size(); ++j)
      new_doflist[j] = doflist[current->_parent_map[j]];
    doflist = new_doflist;
  }
  return doflist;
}
//-----------------------------------------------------------------------------
