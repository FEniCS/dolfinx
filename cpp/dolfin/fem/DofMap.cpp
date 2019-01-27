// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMap.h"
#include "DofMapBuilder.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <ufc.h>
#include <unordered_map>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ufc_dofmap> ufc_dofmap,
               const mesh::Mesh& mesh)
    : _cell_dimension(-1), _ufc_dofmap(ufc_dofmap), _global_dimension(0),
      _ufc_offset(-1)
{
  assert(_ufc_dofmap);
  _cell_dimension = _ufc_dofmap->num_element_support_dofs
                    + _ufc_dofmap->num_global_support_dofs;

  std::tie(_global_dimension, _index_map, _shared_nodes, _neighbours, _dofmap)
      = DofMapBuilder::build(*_ufc_dofmap, mesh);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& parent_dofmap,
               const std::vector<std::size_t>& component,
               const mesh::Mesh& mesh)
    : _cell_dimension(-1), _global_dimension(-1), _ufc_offset(0),
      _index_map(parent_dofmap._index_map)
{
  // FIXME: large objects could be shared (using std::shared_ptr)
  // between parent and view

  // FIXME: the index map block size will be wrong here???

  // Convenience reference to parent UFC dofmap
  const std::int64_t parent_offset
      = parent_dofmap._ufc_offset > 0 ? parent_dofmap._ufc_offset : 0;

  // Build sub-dofmap
  assert(parent_dofmap._ufc_dofmap);
  std::tie(_ufc_dofmap, _ufc_offset, _global_dimension, _dofmap)
      = DofMapBuilder::build_sub_map_view(
          parent_dofmap, *parent_dofmap._ufc_dofmap, parent_dofmap.block_size(),
          parent_offset, component, mesh);

  assert(_ufc_dofmap);
  _cell_dimension = _ufc_dofmap->num_element_support_dofs
                    + _ufc_dofmap->num_global_support_dofs;

  // FIXME: this will be wrong
  _shared_nodes = parent_dofmap._shared_nodes;

  // FIXME: this set may be larger than it should be, e.g. if subdofmap
  // has only facets dofs and parent included vertex dofs.
  _neighbours = parent_dofmap._neighbours;
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
               const DofMap& dofmap_view, const mesh::Mesh& mesh)
    : _cell_dimension(-1), _ufc_dofmap(dofmap_view._ufc_dofmap),
      _global_dimension(-1), _ufc_offset(-1)
{
  assert(_ufc_dofmap);

  // Check dimensional consistency between UFC dofmap and the mesh
  check_provided_entities(*_ufc_dofmap, mesh);

  // Build new dof map
  std::tie(_global_dimension, _index_map, _shared_nodes, _neighbours, _dofmap)
      = DofMapBuilder::build(*_ufc_dofmap, mesh);
  _cell_dimension = _ufc_dofmap->num_element_support_dofs
                    + _ufc_dofmap->num_global_support_dofs;

  // Dimension sanity checks
  assert(dofmap_view._dofmap.size()
         == (std::size_t)(mesh.num_cells() * dofmap_view._cell_dimension));
  assert(global_dimension() == dofmap_view.global_dimension());
  assert(_dofmap.size() == (std::size_t)(mesh.num_cells() * _cell_dimension));

  // FIXME: Could we use a std::vector instead of std::map if the
  //        collapsed dof map is contiguous (0, . . . , n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (std::int64_t i = 0; i < mesh.num_cells(); ++i)
  {
    auto view_cell_dofs = dofmap_view.cell_dofs(i);
    auto cell_dofs = this->cell_dofs(i);
    assert(view_cell_dofs.size() == cell_dofs.size());

    for (Eigen::Index j = 0; j < view_cell_dofs.size(); ++j)
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
  }
}
//-----------------------------------------------------------------------------
std::int64_t DofMap::global_dimension() const { return _global_dimension; }
//-----------------------------------------------------------------------------
std::size_t DofMap::num_element_dofs(std::size_t cell_index) const
{
  return _cell_dimension;
}
//-----------------------------------------------------------------------------
std::size_t DofMap::max_element_dofs() const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->num_element_support_dofs
         + _ufc_dofmap->num_global_support_dofs;
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_entity_dofs(std::size_t entity_dim) const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->num_entity_dofs[entity_dim];
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_entity_closure_dofs(std::size_t entity_dim) const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->num_entity_closure_dofs[entity_dim];
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> DofMap::ownership_range() const
{
  // assert(_index_map);
  auto block_range = _index_map->local_range();
  std::int64_t bs = _index_map->block_size();
  return {{bs * block_range[0], bs * block_range[1]}};
}
//-----------------------------------------------------------------------------
const std::unordered_map<int, std::vector<int>>& DofMap::shared_nodes() const
{
  return _shared_nodes;
}
//-----------------------------------------------------------------------------
const std::set<int>& DofMap::neighbours() const { return _neighbours; }
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, 1>
DofMap::tabulate_entity_closure_dofs(std::size_t entity_dim,
                                     std::size_t cell_entity_index) const
{
  assert(_ufc_dofmap);
  Eigen::Array<int, Eigen::Dynamic, 1> element_dofs(
      _ufc_dofmap->num_entity_closure_dofs[entity_dim]);
  assert(_ufc_dofmap->tabulate_entity_closure_dofs);
  _ufc_dofmap->tabulate_entity_closure_dofs(element_dofs.data(), entity_dim,
                                            cell_entity_index);
  return element_dofs;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, 1>
DofMap::tabulate_entity_dofs(std::size_t entity_dim,
                             std::size_t cell_entity_index) const
{
  assert(_ufc_dofmap);
  if (_ufc_dofmap->num_entity_dofs[entity_dim] == 0)
    return Eigen::Array<int, Eigen::Dynamic, 1>();

  Eigen::Array<int, Eigen::Dynamic, 1> element_dofs(
      _ufc_dofmap->num_entity_dofs[entity_dim]);
  assert(_ufc_dofmap->tabulate_entity_dofs);
  _ufc_dofmap->tabulate_entity_dofs(element_dofs.data(), entity_dim,
                                    cell_entity_index);
  return element_dofs;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::size_t, Eigen::Dynamic, 1>
DofMap::tabulate_global_dofs() const
{
  assert(_global_nodes.empty() or block_size() == 1);

  Eigen::Array<std::size_t, Eigen::Dynamic, 1> dofs(_global_nodes.size());
  std::size_t i = 0;
  for (auto d : _global_nodes)
    dofs[i++] = d;

  return dofs;
}
//-----------------------------------------------------------------------------
std::unique_ptr<GenericDofMap>
DofMap::extract_sub_dofmap(const std::vector<std::size_t>& component,
                           const mesh::Mesh& mesh) const
{
  return std::unique_ptr<GenericDofMap>(new DofMap(*this, component, mesh));
}
//-----------------------------------------------------------------------------
std::pair<std::shared_ptr<GenericDofMap>,
          std::unordered_map<std::size_t, std::size_t>>
DofMap::collapse(const mesh::Mesh& mesh) const
{
  std::unordered_map<std::size_t, std::size_t> collapsed_map;
  std::shared_ptr<GenericDofMap> dofmap(new DofMap(collapsed_map, *this, mesh));
  return std::make_pair(dofmap, std::move(collapsed_map));
}
//-----------------------------------------------------------------------------
void DofMap::set(Vec x, PetscScalar value) const
{
  assert(x);
  la::VecWrapper _x(x);
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x_array = _x.x;
  for (auto index : _dofmap)
    x_array[index] = value;
}
//-----------------------------------------------------------------------------
void DofMap::check_provided_entities(const ufc_dofmap& dofmap,
                                     const mesh::Mesh& mesh)
{
  // Check that we have all mesh entities
  for (std::size_t d = 0; d <= mesh.topology().dim(); ++d)
  {
    if (dofmap.num_entity_dofs[d] > 0 && mesh.num_entities(d) == 0)
    {
      throw std::runtime_error("Missing entities of dimension "
                               + std::to_string(d) + " in dofmap construction");
    }
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> DofMap::index_map() const
{
  return _index_map;
}
//-----------------------------------------------------------------------------
int DofMap::block_size() const
{
  // FIXME: this will almost always be wrong for a sub-dofmap because
  // it shares the  index map with the  parent.
  return _index_map->block_size();
}
//-----------------------------------------------------------------------------
std::string DofMap::str(bool verbose) const
{
  std::stringstream s;
  s << "<DofMap of global dimension " << global_dimension() << ">" << std::endl;
  if (verbose)
  {
    // Cell loop
    assert(_dofmap.size() % _cell_dimension == 0);
    const std::size_t ncells = _dofmap.size() / _cell_dimension;

    for (std::size_t i = 0; i < ncells; ++i)
    {
      s << "Local cell index, cell dofmap dimension: " << i << ", "
        << _cell_dimension << std::endl;

      // Local dof loop
      for (int j = 0; j < _cell_dimension; ++j)
      {
        s << "  "
          << "Local, global dof indices: " << j << ", "
          << _dofmap[i * _cell_dimension + j] << std::endl;
      }
    }
  }

  return s.str();
}
//-----------------------------------------------------------------------------
