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
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{

// Build a collapsed DofMap from a dofmap view
fem::DofMap build_collapsed_dofmap(const DofMap& dofmap_view,
                                   const mesh::Mesh& mesh)
{
  auto element_dof_layout = std::make_shared<ElementDofLayout>(
      *dofmap_view.element_dof_layout, true);
  assert(element_dof_layout);

  if (dofmap_view.index_map->block_size == 1
      and element_dof_layout->block_size > 1)
  {
    throw std::runtime_error(
        "Cannot collapse dofmap with block size greater "
        "than 1 from parent with block size of 1. Create new dofmap first.");
  }

  if (dofmap_view.index_map->block_size > 1
      and element_dof_layout->block_size > 1)
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
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> cell_dofs
        = dofmap_view.cell_dofs(i);
    for (Eigen::Index dof = 0; dof < cell_dofs.rows(); ++dof)
      dofs_view.push_back(cell_dofs[dof]);
  }
  std::sort(dofs_view.begin(), dofs_view.end());
  dofs_view.erase(std::unique(dofs_view.begin(), dofs_view.end()),
                  dofs_view.end());

  // Get block sizes
  const int bs_view = dofmap_view.index_map->block_size;
  const int bs = element_dof_layout->block_size;

  // Compute sizes
  const std::int32_t num_owned_view = dofmap_view.index_map->size_local();
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
  std::vector<std::int64_t> global_index(dofmap_view.index_map->size_local(),
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
      dofmap_view.index_map->num_ghosts(), -1);
  dofmap_view.index_map->scatter_fwd(global_index, global_index_remote, 1);

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
  auto index_map = std::make_shared<common::IndexMap>(mesh.mpi_comm(),
                                                      num_owned, ghosts, bs);

  // Creat array from dofs in view to new dof indices
  std::vector<std::int32_t> old_to_new(dofs_view.back() + 1, -1);
  PetscInt count = 0;
  for (auto& dof : dofs_view)
    old_to_new[dof] = count++;

  // Build new dofmap
  Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dof_array_view
      = dofmap_view.dof_array();
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _dofmap(dof_array_view.size());
  for (Eigen::Index i = 0; i < _dofmap.size(); ++i)
  {
    PetscInt dof_view = dof_array_view[i];
    _dofmap[i] = old_to_new[dof_view];
  }

  // Dimension sanity checks
  assert(element_dof_layout);
  assert(_dofmap.size()
         == (mesh.num_entities(tdim) * element_dof_layout->num_dofs()));

  return fem::DofMap(element_dof_layout, index_map, _dofmap);
}
} // namespace

//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ElementDofLayout> element_dof_layout,
               std::shared_ptr<const common::IndexMap> index_map,
               const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& dofmap)
    : element_dof_layout(element_dof_layout), index_map(index_map),
      _dofmap(dofmap)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DofMap DofMap::extract_sub_dofmap(const std::vector<int>& component,
                                  const mesh::Mesh& mesh) const
{
  return DofMapBuilder::build_submap(*this, component, mesh);
}
//-----------------------------------------------------------------------------
std::pair<std::unique_ptr<DofMap>, std::vector<PetscInt>>
DofMap::collapse(const mesh::Mesh& mesh) const
{
  assert(element_dof_layout);
  assert(index_map);
  std::unique_ptr<DofMap> dofmap_new;
  if (this->index_map->block_size == 1
      and this->element_dof_layout->block_size > 1)
  {
    // Create new element dof layout and reset parent
    auto collapsed_dof_layout
        = std::make_shared<ElementDofLayout>(*element_dof_layout, true);

    // Parent does not have block structure but sub-map does, so build
    // new submap to get block structure for collapsed dofmap.
    dofmap_new = std::make_unique<DofMap>(
        DofMapBuilder::build(mesh, collapsed_dof_layout));
  }
  else
  {
    // Collapse dof map, without build and re-ordering from scratch
    // dofmap_new = std::shared_ptr<DofMap>(new DofMap(*this, mesh));
    dofmap_new = std::make_unique<DofMap>(build_collapsed_dofmap(*this, mesh));
  }
  assert(dofmap_new);

  // Build map from collapsed dof index to original dof index
  auto index_map_new = dofmap_new->index_map;
  std::int32_t size
      = (index_map_new->size_local() + index_map_new->num_ghosts())
        * index_map_new->block_size;
  std::vector<PetscInt> collapsed_map(size);
  const int tdim = mesh.topology().dim();
  for (std::int64_t c = 0; c < mesh.num_entities(tdim); ++c)
  {
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> view_cell_dofs
        = this->cell_dofs(c);
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> cell_dofs
        = dofmap_new->cell_dofs(c);
    assert(view_cell_dofs.size() == cell_dofs.size());

    for (Eigen::Index j = 0; j < cell_dofs.size(); ++j)
    {
      assert(cell_dofs[j] < (int)collapsed_map.size());
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
    }
  }

  return std::make_pair(std::move(dofmap_new), std::move(collapsed_map));
}
//-----------------------------------------------------------------------------
void DofMap::set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
                 PetscScalar value) const
{
  for (Eigen::Index i = 0; i < _dofmap.rows(); ++i)
    x[_dofmap[i]] = value;
}
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
DofMap::dof_array() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
std::string DofMap::str(bool verbose) const
{
  assert(element_dof_layout);

  std::stringstream s;
  if (element_dof_layout->is_view())
    s << "<DofMap view>" << std::endl;
  else
  {
    assert(index_map);
    s << "<DofMap of global dimension "
      << index_map->size_global() * index_map->block_size << ">" << std::endl;
  }

  if (verbose)
  {
    // Cell loop
    assert(element_dof_layout);
    const int cell_dimension = element_dof_layout->num_dofs();

    assert(_dofmap.size() % cell_dimension == 0);
    const std::int32_t ncells = _dofmap.size() / cell_dimension;
    for (std::int32_t i = 0; i < ncells; ++i)
    {
      s << "Local cell index, cell dofmap dimension: " << i << ", "
        << cell_dimension << std::endl;

      // Local dof loop
      for (int j = 0; j < cell_dimension; ++j)
      {
        s << "  "
          << "Local, global dof indices: " << j << ", "
          << _dofmap[i * cell_dimension + j] << std::endl;
      }
    }
  }

  return s.str();
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 1> DofMap::dofs(const mesh::Mesh& mesh,
                                                       std::size_t dim) const
{
  assert(element_dof_layout);

  // Check number of dofs per entity (on each cell)
  const int num_dofs_per_entity = element_dof_layout->num_entity_dofs(dim);

  // Return empty vector if not dofs on requested entity
  if (num_dofs_per_entity == 0)
    return Eigen::Array<PetscInt, Eigen::Dynamic, 1>();

  // Vector to hold list of dofs
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dof_list(mesh.num_entities(dim)
                                                     * num_dofs_per_entity);

  // Build local dofs for each entity of dimension dim
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> entity_dofs_local;
  for (int i = 0; i < mesh::cell_num_entities(mesh.cell_type, dim); ++i)
    entity_dofs_local.push_back(element_dof_layout->entity_dofs(dim, i));

  // Iterate over cells
  const int tdim = mesh.topology().dim();
  for (auto& c : mesh::MeshRange(mesh, tdim))
  {
    // Get local-to-global dofmap for cell
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> cell_dof_list
        = cell_dofs(c.index());

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
