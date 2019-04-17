// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMap.h"
#include "DofMapBuilder.h"
#include "ElementDofLayout.h"
#include "utils.h"
#include <cstdint>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <unordered_map>

#include <boost/timer/timer.hpp>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
DofMap::DofMap(const ufc_dofmap& ufc_dofmap, const mesh::Mesh& mesh)
    : DofMap(std::make_shared<ElementDofLayout>(
                 create_element_dof_layout(ufc_dofmap, {}, mesh.type())),
             mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ElementDofLayout> element_dof_layout,
               const mesh::Mesh& mesh)
    : _cell_dimension(element_dof_layout->num_dofs()), _global_dimension(-1),
      _element_dof_layout(element_dof_layout)
{
  const int bs = _element_dof_layout->block_size();
  if (bs == 1)
  {
    std::tie(_global_dimension, _index_map, _shared_nodes, _dofmap)
        = DofMapBuilder::build(mesh, *_element_dof_layout, bs);
  }
  else
  {
    std::tie(_global_dimension, _index_map, _shared_nodes, _dofmap)
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
  // FIXME: Large objects could be shared (using std::shared_ptr)
  //        between parent and view

  assert(!component.empty());
  const int D = mesh.topology().dim();

  // Set element dof layout and cell dimension
  _element_dof_layout
      = dofmap_parent._element_dof_layout->sub_dofmap(component);
  _cell_dimension = _element_dof_layout->num_dofs();

  // Get components in parent map that correspond to sub-dofs
  assert(dofmap_parent._element_dof_layout);
  const std::vector<int> element_map_view
      = dofmap_parent._element_dof_layout->sub_view(component);

  // Build dofmap by extracting from parent
  const std::int32_t dofs_per_cell = element_map_view.size();
  _dofmap.resize(dofs_per_cell * mesh.num_entities(D));
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    const int c = cell.index();
    auto cell_dmap_parent = dofmap_parent.cell_dofs(c);
    for (std::int32_t i = 0; i < dofs_per_cell; ++i)
      _dofmap[c * dofs_per_cell + i] = cell_dmap_parent[element_map_view[i]];
  }

  // Compute global dimension of sub-map
  _global_dimension = 0;
  for (int d = 0; d <= D; ++d)
  {
    const std::int64_t n = mesh.num_entities_global(d);
    _global_dimension += n * _element_dof_layout->num_entity_dofs(d);
  }

  // FIXME: This stores more than is required. Compress, or share with
  // parent.
  _shared_nodes = dofmap_parent._shared_nodes;
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& dofmap_view, const mesh::Mesh& mesh)
    : _cell_dimension(dofmap_view._element_dof_layout->num_dofs()),
      _global_dimension(dofmap_view._global_dimension),
      _element_dof_layout(dofmap_view._element_dof_layout)
{
  if (dofmap_view._index_map->block_size() == 1
      and dofmap_view._element_dof_layout->block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot collapse dofmap with block size greater "
        "than 1 from parent with block size of 1. Create new dofmap first.");
  }

  if (dofmap_view._index_map->block_size() > 1
      and dofmap_view._element_dof_layout->block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot (yet) collapse dofmap with block size greater "
        "than 1 from parent with block size greater than 1. Create new dofmap "
        "first.");
  }

  if (dofmap_view._index_map->block_size() != 1)
    throw std::runtime_error("Block size greater than 1 not supported yet.");

  boost::timer::auto_cpu_timer t;

  // Get topological dimension
  const int tdim = mesh.topology().dim();

  // Build set of dofs that are in the new dofmap
  std::vector<std::int32_t> dofs_view;
  for (std::int64_t i = 0; i < mesh.num_entities(tdim); ++i)
  {
    auto cell_dofs = dofmap_view.cell_dofs(i);
    for (Eigen::Index dof = 0; dof < cell_dofs.rows(); ++dof)
      dofs_view.push_back(cell_dofs[dof]);
  }
  std::sort(dofs_view.begin(), dofs_view.end());
  dofs_view.erase(std::unique(dofs_view.begin(), dofs_view.end()),
                  dofs_view.end());

  // Compute sizes
  const std::int32_t num_owned_view = dofmap_view._index_map->size_local();
  const auto it_unowned0
      = std::lower_bound(dofs_view.begin(), dofs_view.end(), num_owned_view);
  const std::int64_t num_owned_new
      = std::distance(dofs_view.begin(), it_unowned0);
  const std::int64_t num_unowned_new
      = std::distance(it_unowned0, dofs_view.end());

  // Get process offset for new dofmap
  const std::int64_t process_offset
      = dolfin::MPI::global_offset(mesh.mpi_comm(), num_owned_new, true);

  // For owned dofs, compute new global index
  std::vector<std::int64_t> global_index_new(
      dofmap_view._index_map->size_local(), -1);
  for (auto it = dofs_view.begin(); it != it_unowned0; ++it)
  {
    const std::int64_t pos = std::distance(dofs_view.begin(), it);
    global_index_new[*it] = pos + process_offset;
  }

  // Send new global indices for owned dofs to non-owning process, and
  // receive new global indices from owner
  std::vector<std::int64_t> global_index_new_remote(
      dofmap_view._index_map->num_ghosts(), -1);
  dofmap_view._index_map->scatter_fwd(global_index_new,
                                      global_index_new_remote);

  // Compute ghosts for collapsed dofmap
  std::vector<std::int64_t> ghosts_new(num_unowned_new);
  for (auto it = it_unowned0; it != dofs_view.end(); ++it)
  {
    const std::int32_t index = std::distance(it_unowned0, it);
    const std::int32_t index_old = *it - num_owned_view;
    ghosts_new[index] = global_index_new_remote[index_old];
  }

  // Create new index map
  _index_map = std::make_shared<common::IndexMap>(mesh.mpi_comm(),
                                                  num_owned_new, ghosts_new, 1);

  // Creat array from dofs in view to new dof indices
  std::vector<std::int32_t> old_to_new(*dofs_view.rbegin() + 1, -1);
  PetscInt count = 0;
  for (auto& dof : dofs_view)
    old_to_new[dof] = count++;

  // Build new dofmap
  _dofmap.resize(dofmap_view._dofmap.size());
  for (std::size_t i = 0; i < _dofmap.size(); ++i)
  {
    PetscInt dof_view = dofmap_view._dofmap[i];
    _dofmap[i] = old_to_new[dof_view];
  }

  // FIXME:
  // Set shared nodes

  // Dimension sanity checks
  assert(
      dofmap_view._dofmap.size()
      == (std::size_t)(mesh.num_entities(tdim) * dofmap_view._cell_dimension));
  assert(global_dimension() == dofmap_view.global_dimension());
  assert(_dofmap.size()
         == (std::size_t)(mesh.num_entities(tdim) * _cell_dimension));
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
  std::shared_ptr<GenericDofMap> dofmap;
  if (this->_index_map->block_size() == 1
      and this->_element_dof_layout->block_size() > 1)
  {
    // Parent does not have block structure but sub-map does, so build new
    // submap to get block structure.
    dofmap = std::shared_ptr<GenericDofMap>(
        new DofMap(this->_element_dof_layout, mesh));
  }
  else
    dofmap = std::shared_ptr<GenericDofMap>(new DofMap(*this, mesh));

  // FIXME: Could we use a std::vector instead of std::map if the
  //        collapsed dof map is contiguous (0, . . . , n)?

  // Build map from collapsed dof index to original dof index
  std::unordered_map<std::size_t, std::size_t> collapsed_map;
  const int tdim = mesh.topology().dim();
  for (std::int64_t i = 0; i < mesh.num_entities(tdim); ++i)
  {
    auto view_cell_dofs = this->cell_dofs(i);
    auto cell_dofs = dofmap->cell_dofs(i);
    assert(view_cell_dofs.size() == cell_dofs.size());

    for (Eigen::Index j = 0; j < view_cell_dofs.size(); ++j)
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
  }

  return std::make_pair(dofmap, std::move(collapsed_map));
}
//-----------------------------------------------------------------------------
void DofMap::set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
                 PetscScalar value) const
{
  for (auto index : _dofmap)
    x[index] = value;
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
