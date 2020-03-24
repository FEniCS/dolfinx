// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ElementDofLayout.h"
#include <array>
#include <dolfinx/common/log.h>
#include <map>
#include <numeric>
#include <set>

#include <iostream>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
ElementDofLayout::ElementDofLayout(
    int block_size, const std::vector<std::vector<std::set<int>>>& entity_dofs,
    const std::vector<int>& parent_map,
    const std::vector<std::shared_ptr<const ElementDofLayout>>& sub_dofmaps,
    const mesh::CellType cell_type,
    const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        base_permutations)
    : _block_size(block_size), _cell_type(cell_type), _parent_map(parent_map),
      _num_dofs(0), _entity_dofs(entity_dofs), _sub_dofmaps(sub_dofmaps),
      _base_permutations(base_permutations)
{
  // TODO: Add size check on base_permutations. Size should be:
  // number of rows = num_edges + 2*num_faces + 4*num_volumes
  // number of columns = number of dofs

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
    assert(dim < (int)entity_dofs.size());
    assert(index < (int)entity_dofs[dim].size());
    int subdim = 0;
    for (auto sub_entity : entity.second)
    {
      assert(subdim < (int)entity_dofs.size());
      for (auto sub_index : sub_entity)
      {
        assert(sub_index < (int)entity_dofs[subdim].size());
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

  // Check that base_permutations has the correct shape
  int perm_count = 0;
  const std::array<int, 4> perms_per_dim = {0, 1, 2, 4};
  for (std::size_t dim = 0; dim < entity_dofs.size() - 1; ++dim)
  {
    assert(dim < perms_per_dim.size());
    assert(dim < entity_dofs.size());
    perm_count += perms_per_dim[dim] * entity_dofs[dim].size();
  }
  if (base_permutations.rows() != perm_count
      or _base_permutations.cols() != _num_dofs)
  {
    throw std::runtime_error("Permutation array has wrong shape. Expected "
                             + std::to_string(perm_count) + " x "
                             + std::to_string(_num_dofs) + " but got "
                             + std::to_string(_base_permutations.rows()) + " x "
                             + std::to_string(_base_permutations.cols()) + ".");
  }
}
//-----------------------------------------------------------------------------
ElementDofLayout ElementDofLayout::copy() const
{
  ElementDofLayout layout(*this);
  layout._parent_map.clear();
  return layout;
}
//-----------------------------------------------------------------------------
mesh::CellType ElementDofLayout::cell_type() const { return _cell_type; }
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
    for (std::size_t j = 0; j < dof_list_new.size(); ++j)
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
int ElementDofLayout::degree() const
{
  switch (_cell_type)
  {
  case mesh::CellType::interval:
    if (num_dofs() == 2)
      return 1;
    else if (num_dofs() == 3)
      return 2;
    break;
  case mesh::CellType::triangle:
    if (num_dofs() == 3)
      return 1;
    else if (num_dofs() == 6)
      return 2;
    break;
  case mesh::CellType::quadrilateral:
    if (num_dofs() == 4)
      return 1;
    else if (num_dofs() == 9)
      return 2;
    break;
  case mesh::CellType::tetrahedron:
    if (num_dofs() == 4)
      return 1;
    else if (num_dofs() == 10)
      return 2;
    break;
  case mesh::CellType::hexahedron:
    if (num_dofs() == 8)
      return 1;
    else if (num_dofs() == 27)
      return 2;
    break;
  default:
    throw std::runtime_error("Unknown cell type");
  }

  throw std::runtime_error("Cannot determine degree");
}
//-----------------------------------------------------------------------------
ElementDofLayout fem::geometry_layout(mesh::CellType cell, int num_nodes)
{
  // TODO: Fix for degree > 2

  const int dim = mesh::cell_dim(cell);
  int num_perms = 0;
  const std::array<int, 4> p_per_dim = {0, 1, 2, 4};
  for (int d = 1; d < dim; ++d)
    num_perms += p_per_dim[d] * mesh::cell_num_entities(cell, d);

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> perm(
      num_perms, num_nodes);
  for (int i = 0; i < perm.rows(); ++i)
    for (int j = 0; j < perm.cols(); ++j)
      perm(i, j) = j;

  // entity_dofs = [[set([0]), set([1]), set([2])], 3 * [set()], [set()]]
  int dof = 0;
  std::vector<std::vector<std::set<int>>> entity_dofs(dim + 1);
  for (int d = 0; d <= dim; ++d)
  {
    const int num_entities = mesh::cell_num_entities(cell, d);
    if (dof < num_nodes)
    {
      for (int e = 0; e < num_entities; ++e)
        entity_dofs[d].push_back({dof++});
    }
    else
    {
      for (int e = 0; e < num_entities; ++e)
        entity_dofs[d].push_back({});
    }
  }

  return fem::ElementDofLayout(1, entity_dofs, {}, {}, cell, perm);
}
//-----------------------------------------------------------------------------
