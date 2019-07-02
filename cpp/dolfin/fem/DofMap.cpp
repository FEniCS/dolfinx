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
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>

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
    std::tie(_global_dimension, _index_map, _dofmap)
        = DofMapBuilder::build(mesh, *_element_dof_layout, bs);
  }
  else
  {
    std::tie(_global_dimension, _index_map, _dofmap)
        = DofMapBuilder::build(mesh, *_element_dof_layout->sub_dofmap({0}), bs);
  }
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& dofmap_parent, const std::vector<int>& component,
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
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& dofmap_view, const mesh::Mesh& mesh)
    : _cell_dimension(dofmap_view._element_dof_layout->num_dofs()),
      _global_dimension(dofmap_view._global_dimension),
      _element_dof_layout(
          new ElementDofLayout(*dofmap_view._element_dof_layout, true))
{
  if (dofmap_view._index_map->block_size() == 1
      and _element_dof_layout->block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot collapse dofmap with block size greater "
        "than 1 from parent with block size of 1. Create new dofmap first.");
  }

  if (dofmap_view._index_map->block_size() > 1
      and _element_dof_layout->block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot (yet) collapse dofmap with block size greater "
        "than 1 from parent with block size greater than 1. Create new dofmap "
        "first.");
  }

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

  // Get block sizes
  const int bs_view = dofmap_view._index_map->block_size();
  const int bs = _element_dof_layout->block_size();

  // Compute sizes
  const std::int32_t num_owned_view = dofmap_view._index_map->size_local();
  const auto it_unowned0 = std::lower_bound(dofs_view.begin(), dofs_view.end(),
                                            num_owned_view * bs_view);
  const std::int64_t num_owned
      = std::distance(dofs_view.begin(), it_unowned0) / bs;
  assert(std::distance(dofs_view.begin(), it_unowned0) % bs == 0);

  const std::int64_t num_unowned
      = std::distance(it_unowned0, dofs_view.end()) / bs;
  assert(std::distance(it_unowned0, dofs_view.end()) % bs == 0);

  // Get process offset for new dofmap
  const std::int64_t process_offset
      = dolfin::MPI::global_offset(mesh.mpi_comm(), num_owned, true);

  // For owned dofs, compute new global index
  std::vector<std::int64_t> global_index(dofmap_view._index_map->size_local(),
                                         -1);
  for (auto it = dofs_view.begin(); it != it_unowned0; ++it)
  {
    const std::int64_t block = std::distance(dofs_view.begin(), it) / bs;
    const std::int32_t block_parent = *it / bs_view;
    global_index[block_parent] = block + process_offset;
  }

  // Send new global indices for owned dofs to non-owning process, and
  // receive new global indices from owner
  std::vector<std::int64_t> global_index_remote(
      dofmap_view._index_map->num_ghosts(), -1);
  dofmap_view._index_map->scatter_fwd(global_index, global_index_remote, 1);

  // Compute ghosts for collapsed dofmap
  std::vector<std::int64_t> ghosts(num_unowned);
  for (auto it = it_unowned0; it != dofs_view.end(); ++it)
  {
    const std::int32_t index = std::distance(it_unowned0, it) / bs;
    const std::int32_t index_old = *it / bs_view - num_owned_view;
    assert(global_index_remote[index_old] >= 0);
    ghosts[index] = global_index_remote[index_old];
  }

  // Create new index map
  _index_map = std::make_shared<common::IndexMap>(mesh.mpi_comm(), num_owned,
                                                  ghosts, bs);

  // Creat array from dofs in view to new dof indices
  std::vector<std::int32_t> old_to_new(dofs_view.back() + 1, -1);
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

  // Dimension sanity checks
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
  assert(_global_nodes.empty() or _index_map->block_size() == 1);
  Eigen::Array<std::size_t, Eigen::Dynamic, 1> dofs(_global_nodes.size());
  std::size_t i = 0;
  for (auto d : _global_nodes)
    dofs[i++] = d;
  return dofs;
}
//-----------------------------------------------------------------------------
std::unique_ptr<DofMap>
DofMap::extract_sub_dofmap(const std::vector<int>& component,
                           const mesh::Mesh& mesh) const
{
  return std::unique_ptr<DofMap>(new DofMap(*this, component, mesh));
}
//-----------------------------------------------------------------------------
std::pair<std::shared_ptr<DofMap>, std::vector<PetscInt>>
DofMap::collapse(const mesh::Mesh& mesh) const
{
  assert(_element_dof_layout);
  assert(_index_map);
  std::shared_ptr<DofMap> dofmap_new;
  if (this->_index_map->block_size() == 1
      and this->_element_dof_layout->block_size() > 1)
  {
    // Create new element dof layout and reset parent
    auto collapsed_dof_layout
        = std::make_shared<ElementDofLayout>(*_element_dof_layout, true);

    // Parent does not have block structure but sub-map does, so build
    // new submap to get block structure for collapsed dofmap.
    dofmap_new = std::make_shared<DofMap>(collapsed_dof_layout, mesh);
  }
  else
  {
    // Collapse dof map without build and re-ordering from scratch
    dofmap_new = std::shared_ptr<DofMap>(new DofMap(*this, mesh));
  }
  assert(dofmap_new);

  // Build map from collapsed dof index to original dof index
  auto index_map_new = dofmap_new->index_map();
  std::int32_t size
      = (index_map_new->size_local() + index_map_new->num_ghosts())
        * index_map_new->block_size();
  std::vector<PetscInt> collapsed_map(size);
  const int tdim = mesh.topology().dim();
  for (std::int64_t c = 0; c < mesh.num_entities(tdim); ++c)
  {
    const auto view_cell_dofs = this->cell_dofs(c);
    const auto cell_dofs = dofmap_new->cell_dofs(c);
    assert(view_cell_dofs.size() == cell_dofs.size());

    for (Eigen::Index j = 0; j < cell_dofs.size(); ++j)
    {
      assert(cell_dofs[j] < (int)collapsed_map.size());
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
    }
  }

  return std::make_pair(dofmap_new, std::move(collapsed_map));
}
//-----------------------------------------------------------------------------
void DofMap::set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
                 PetscScalar value) const
{
  for (auto index : _dofmap)
    x[index] = value;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> DofMap::index_map() const
{
  return _index_map;
}
//-----------------------------------------------------------------------------
Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
DofMap::dof_array() const
{
  return Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(
      _dofmap.data(), _dofmap.size());
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
Eigen::Array<std::size_t, Eigen::Dynamic, 1>
DofMap::tabulate_local_to_global_dofs() const
{
  // FIXME: use common::IndexMap::local_to_global_index?

  const auto idxmap = index_map();
  assert(idxmap);
  const std::size_t bs = idxmap->block_size();
  const auto& local_to_global_unowned = idxmap->ghosts();
  const std::size_t local_ownership_size = bs * idxmap->size_local();

  Eigen::Array<std::size_t, Eigen::Dynamic, 1> local_to_global_map(
      bs * (idxmap->size_local() + idxmap->num_ghosts()));

  const std::size_t global_offset = bs * idxmap->local_range()[0];
  for (std::size_t i = 0; i < local_ownership_size; ++i)
    local_to_global_map[i] = i + global_offset;

  for (Eigen::Index node = 0; node < local_to_global_unowned.size(); ++node)
  {
    for (std::size_t component = 0; component < bs; ++component)
    {
      local_to_global_map[bs * node + component + local_ownership_size]
          = bs * local_to_global_unowned[node] + component;
    }
  }

  return local_to_global_map;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 1> DofMap::dofs(const mesh::Mesh& mesh,
                                                       std::size_t dim) const
{
  // Check number of dofs per entity (on each cell)
  const std::size_t num_dofs_per_entity = num_entity_dofs(dim);

  // Return empty vector if not dofs on requested entity
  if (num_dofs_per_entity == 0)
    return Eigen::Array<PetscInt, Eigen::Dynamic, 1>();

  // Vector to hold list of dofs
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dof_list(mesh.num_entities(dim)
                                                     * num_dofs_per_entity);

  // Build local dofs for each entity of dimension dim
  const mesh::CellType& cell_type = mesh.type();
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> entity_dofs_local;
  for (std::size_t i = 0; i < cell_type.num_entities(dim); ++i)
    entity_dofs_local.push_back(tabulate_entity_dofs(dim, i));

  // Iterate over cells
  for (auto& c : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Get local-to-global dofmap for cell
    const auto cell_dof_list = cell_dofs(c.index());

    // Loop over all entities of dimension dim
    unsigned int local_index = 0;
    for (auto& e : mesh::EntityRange<mesh::MeshEntity>(c, dim))
    {
      // Get dof index and add to list
      for (Eigen::Index i = 0; i < entity_dofs_local[local_index].size(); ++i)
      {
        const std::size_t entity_dof_local = entity_dofs_local[local_index][i];
        const PetscInt dof_index = cell_dof_list[entity_dof_local];
        assert((Eigen::Index)(e.index() * num_dofs_per_entity + i)
               < dof_list.size());
        dof_list[e.index() * num_dofs_per_entity + i] = dof_index;
      }
      ++local_index;
    }
  }

  return dof_list;
}
//-----------------------------------------------------------------------------
