// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMap.h"
#include "DofMapBuilder.h"
#include "ElementDofLayout.h"
#include "utils.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <unordered_map>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
DofMap::DofMap(const ufc_dofmap& ufc_dofmap, const mesh::Mesh& mesh)
    : _cell_dimension(-1), _global_dimension(0)
{
  _element_dof_layout = std::make_shared<ElementDofLayout>(
      create_element_dof_layout(ufc_dofmap, {}, mesh.type()));
  _cell_dimension = _element_dof_layout->num_dofs();

  const int bs = _element_dof_layout->block_size();
  if (bs == 1)
  {
    std::tie(_global_dimension, _index_map, _shared_nodes, _neighbours, _dofmap)
        = DofMapBuilder::build(mesh, *_element_dof_layout, bs);
  }
  else
  {
    std::tie(_global_dimension, _index_map, _shared_nodes, _neighbours, _dofmap)
        = DofMapBuilder::build(mesh, *_element_dof_layout->sub_dofmap({0}), bs);
  }
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& dofmap_parent,
               const std::vector<std::size_t>& component,
               const mesh::Mesh& mesh)
    : _cell_dimension(-1), _global_dimension(-1),
      _index_map(dofmap_parent._index_map)
{
  // FIXME: large objects could be shared (using std::shared_ptr)
  // between parent and view

  // FIXME: the index map block size will likely be wrong?

  std::shared_ptr<const ElementDofLayout> element_dof_layout_parent(
      dofmap_parent._element_dof_layout);

  _element_dof_layout = element_dof_layout_parent->sub_dofmap(component);
  _cell_dimension = _element_dof_layout->num_dofs();

  // Build sub-dofmap
  std::tie(_global_dimension, _dofmap) = DofMapBuilder::build_sub_map_view(
      dofmap_parent, *element_dof_layout_parent, component, mesh);

  // FIXME: This stores more than is required. Compress, or share with
  // parent.
  _shared_nodes = dofmap_parent._shared_nodes;

  // FIXME: this set may be larger than it should be, e.g. if subdofmap
  // has only facets dofs and parent included vertex dofs.
  _neighbours = dofmap_parent._neighbours;
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
               const DofMap& dofmap_view, const mesh::Mesh& mesh)
    : _cell_dimension(-1), _global_dimension(-1),
      _element_dof_layout(dofmap_view._element_dof_layout)
{
  _cell_dimension = _element_dof_layout->num_dofs();

  // Check dimensional consistency between ElementDofLayout and the mesh
  check_provided_entities(*_element_dof_layout, mesh);

  // Build new dof map
  const int bs = _element_dof_layout->block_size();
  if (bs == 1)
  {
    std::tie(_global_dimension, _index_map, _shared_nodes, _neighbours, _dofmap)
        = DofMapBuilder::build(mesh, *_element_dof_layout, bs);
  }
  else
  {
    std::tie(_global_dimension, _index_map, _shared_nodes, _neighbours, _dofmap)
        = DofMapBuilder::build(mesh, *_element_dof_layout->sub_dofmap({0}), bs);
  }

  const int tdim = mesh.topology().dim();

  // Dimension sanity checks
  assert(
      dofmap_view._dofmap.size()
      == (std::size_t)(mesh.num_entities(tdim) * dofmap_view._cell_dimension));
  assert(global_dimension() == dofmap_view.global_dimension());
  assert(_dofmap.size()
         == (std::size_t)(mesh.num_entities(tdim) * _cell_dimension));

  // FIXME: Could we use a std::vector instead of std::map if the
  //        collapsed dof map is contiguous (0, . . . , n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (std::int64_t i = 0; i < mesh.num_entities(tdim); ++i)
  {
    auto view_cell_dofs = dofmap_view.cell_dofs(i);
    auto cell_dofs = this->cell_dofs(i);
    assert(view_cell_dofs.size() == cell_dofs.size());

    for (Eigen::Index j = 0; j < view_cell_dofs.size(); ++j)
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
  }
}
//-----------------------------------------------------------------------------
bool DofMap::is_view() const
{
  assert(_element_dof_layout);
  return _element_dof_layout->is_view();
}
//-----------------------------------------------------------------------------
std::int64_t DofMap::global_dimension() const { return _global_dimension; }
//-----------------------------------------------------------------------------
std::size_t DofMap::num_element_dofs(std::size_t cell_index) const
{
  return _cell_dimension;
}
//-----------------------------------------------------------------------------
std::size_t DofMap::max_element_dofs() const { return _cell_dimension; }
//-----------------------------------------------------------------------------
std::size_t DofMap::num_entity_dofs(std::size_t entity_dim) const
{
  assert(_element_dof_layout);
  return _element_dof_layout->num_entity_dofs(entity_dim);
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_entity_closure_dofs(std::size_t entity_dim) const
{
  assert(_element_dof_layout);
  return _element_dof_layout->num_entity_closure_dofs(entity_dim);
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
  const std::vector<std::vector<std::set<int>>>& dofs
      = _element_dof_layout->entity_closure_dofs();
  assert(entity_dim < dofs.size());
  assert(cell_entity_index < dofs[entity_dim].size());
  Eigen::Array<int, Eigen::Dynamic, 1> element_dofs(
      dofs[entity_dim][cell_entity_index].size());
  std::copy(dofs[entity_dim][cell_entity_index].begin(),
            dofs[entity_dim][cell_entity_index].end(), element_dofs.data());
  return element_dofs;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, 1>
DofMap::tabulate_entity_dofs(std::size_t entity_dim,
                             std::size_t cell_entity_index) const
{
  const std::vector<std::vector<std::set<int>>>& dofs
      = _element_dof_layout->entity_dofs();
  assert(entity_dim < dofs.size());
  assert(cell_entity_index < dofs[entity_dim].size());
  Eigen::Array<int, Eigen::Dynamic, 1> element_dofs(
      dofs[entity_dim][cell_entity_index].size());
  std::copy(dofs[entity_dim][cell_entity_index].begin(),
            dofs[entity_dim][cell_entity_index].end(), element_dofs.data());
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
void DofMap::check_provided_entities(const ElementDofLayout& dofmap,
                                     const mesh::Mesh& mesh)
{
  // Check that we have all mesh entities
  for (int d = 0; d <= mesh.topology().dim(); ++d)
  {
    if (dofmap.num_entity_dofs(d) > 0 && mesh.num_entities(d) == 0)
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
