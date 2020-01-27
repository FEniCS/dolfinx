// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DirichletBC.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <dolfinx/mesh/cell_types.h>
#include <map>
#include <utility>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
std::vector<std::array<PetscInt, 2>>
get_remote_bcs(const common::IndexMap& map, const common::IndexMap& map_g,
               const std::vector<std::array<PetscInt, 2>>& dofs_local)
{
  std::vector<std::array<PetscInt, 2>> dof_dof_g;

  const std::int32_t bs = map.block_size;
  const std::int32_t size_owned = map.size_local();
  const std::int32_t size_ghost = map.num_ghosts();

  const std::int32_t bs_g = map_g.block_size;
  const std::int32_t size_owned_g = map_g.size_local();
  const std::int32_t size_ghost_g = map_g.num_ghosts();
  const std::array<std::int64_t, 2> range_g = map_g.local_range();
  const std::int64_t offset_g = range_g[0];

  // For each dof local index, store global index in Vg (-1 if no bc)
  std::vector<PetscInt> marker_owned(bs * size_owned, -1);
  std::vector<PetscInt> marker_ghost(bs * size_ghost, -1);
  for (auto& dofs : dofs_local)
  {
    const PetscInt index_block_g = dofs[1] / bs_g;
    const PetscInt pos_g = dofs[1] % bs_g;
    if (dofs[0] < bs * size_owned)
    {
      marker_owned[dofs[0]]
          = bs_g * map_g.local_to_global(index_block_g) + pos_g;
    }
    else
    {
      marker_ghost[dofs[0] - (bs * size_owned)]
          = bs_g * map_g.local_to_global(index_block_g) + pos_g;
    }
  }

  // Build global-to-local map for ghost indices (blocks) in map_g
  std::map<PetscInt, PetscInt> global_to_local_g;
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghosts_g = map_g.ghosts();
  for (Eigen::Index i = 0; i < size_owned_g; ++i)
    global_to_local_g.insert({i + offset_g, i});
  for (Eigen::Index i = 0; i < size_ghost_g; ++i)
    global_to_local_g.insert({ghosts_g[i], i + size_owned_g});

  // For each owned bc index, scatter associated g global index to ghost
  // processes
  std::vector<PetscInt> marker_ghost_rcvd = map.scatter_fwd(marker_owned, bs);
  assert((int)marker_ghost_rcvd.size() == size_ghost * bs);

  // Add to (local index)-(local g index) map
  for (std::size_t i = 0; i < marker_ghost_rcvd.size(); ++i)
  {
    if (marker_ghost_rcvd[i] > -1)
    {
      const PetscInt index_block_g = marker_ghost_rcvd[i] / bs_g;
      const PetscInt pos_g = marker_ghost_rcvd[i] % bs_g;
      const auto it = global_to_local_g.find(index_block_g);
      assert(it != global_to_local_g.end());
      dof_dof_g.push_back(
          {(PetscInt)(bs * size_owned + i), bs_g * it->second + pos_g});
    }
  }

  // Scatter (reverse) data from ghost processes to owner
  std::vector<PetscInt> marker_owner_rcvd(bs * size_owned, -1);
  map.scatter_rev(marker_owner_rcvd, marker_ghost, bs,
                  common::IndexMap::Mode::insert);
  assert((int)marker_owner_rcvd.size() == size_owned * bs);
  for (std::size_t i = 0; i < marker_owner_rcvd.size(); ++i)
  {
    if (marker_owner_rcvd[i] >= 0)
    {
      const PetscInt index_global_g = marker_owner_rcvd[i];
      const PetscInt index_block_g = index_global_g / bs_g;
      const PetscInt pos_g = index_global_g % bs_g;
      const auto it = global_to_local_g.find(index_block_g);
      assert(it != global_to_local_g.end());
      dof_dof_g.push_back({(PetscInt)i, bs_g * it->second + pos_g});
    }
  }

  return dof_dof_g;
}
} // namespace
//-----------------------------------------------------------------------------
std::vector<PetscInt>
get_remote_bcs(const common::IndexMap& map,
               const std::vector<PetscInt>& dofs_local)
{
  const std::int32_t bs = map.block_size;
  const std::int32_t size_owned = map.size_local();
  const std::int32_t size_ghost = map.num_ghosts();

  const std::array<std::int64_t, 2> range = map.local_range();
  const std::int64_t offset = range[0];

  // For each dof local index, store global index (-1 if no bc)
  std::vector<PetscInt> marker_owned(bs * size_owned, -1);
  std::vector<PetscInt> marker_ghost(bs * size_ghost, -1);
  for (auto& dofs : dofs_local)
  {
    const PetscInt index_block = dofs / bs;
    const PetscInt pos = dofs % bs;
    if (dofs < bs * size_owned)
    {
      marker_owned[dofs]
          = bs * map.local_to_global(index_block) + pos;
    }
    else
    {
      marker_ghost[dofs - (bs * size_owned)]
          = bs * map.local_to_global(index_block) + pos;
    }
  }

  // Build global-to-local map for ghost indices (blocks) in map
  std::map<PetscInt, PetscInt> global_to_local;
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghosts = map.ghosts();
  for (Eigen::Index i = 0; i < size_owned; ++i)
    global_to_local.insert({i + offset, i});
  for (Eigen::Index i = 0; i < size_ghost; ++i)
    global_to_local.insert({ghosts[i], i + size_owned});

  // For each owned bc index, scatter associated global index to ghost
  // processes
  std::vector<PetscInt> marker_ghost_rcvd = map.scatter_fwd(marker_owned, bs);
  assert((int)marker_ghost_rcvd.size() == size_ghost * bs);

  std::vector<PetscInt> dofs;

  // Add to local indices map
  for (std::size_t i = 0; i < marker_ghost_rcvd.size(); ++i)
  {
    if (marker_ghost_rcvd[i] > -1)
    {
      const PetscInt index_block = marker_ghost_rcvd[i] / bs;
      const auto it = global_to_local.find(index_block);
      assert(it != global_to_local.end());
      dofs.push_back((PetscInt)(bs * size_owned + i));
    }
  }

  return dofs;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 2> fem::locate_dofs_topological(
    const function::FunctionSpace& V0, const int dim,
    const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& entities,
    const function::FunctionSpace& V1, bool remote)
{
  // Get mesh
  assert(V0.mesh());
  assert(V1.mesh());
  if (V0.mesh() != V1.mesh())
    throw std::runtime_error("Meshes are not the same.");
  const mesh::Mesh& mesh = *V0.mesh();
  const std::size_t tdim = mesh.topology().dim();

  assert(V0.element());
  assert(V1.element());

  if (!V0.has_element(*V1.element()))
  {
    throw std::runtime_error("Function spaces must have the same elements or "
                             "one be a subelement of another.");
  }

  // Get dofmaps
  assert(V0.dofmap());
  assert(V1.dofmap());
  const DofMap& dofmap0 = *V0.dofmap();
  const DofMap& dofmap1 = *V1.dofmap();

  // Initialise entity-cell connectivity
  mesh.create_entities(tdim);
  mesh.create_connectivity(dim, tdim);

  // Allocate space
  assert(dofmap0.element_dof_layout);
  const int num_entity_dofs
      = dofmap0.element_dof_layout->num_entity_closure_dofs(dim);

  // Build vector local dofs for each cell facet
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> entity_dofs;
  for (int i = 0; i < mesh::cell_num_entities(mesh.cell_type(), dim); ++i)
  {
    entity_dofs.push_back(
        dofmap0.element_dof_layout->entity_closure_dofs(dim, i));
  }

  // Iterate over marked facets
  std::vector<std::array<PetscInt, 2>> bc_dofs;
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    // Create facet and attached cell
    const mesh::MeshEntity entity(mesh, dim, entities[e]);
    const std::size_t cell_index = entity.entities(tdim)[0];
    const mesh::MeshEntity cell(mesh, tdim, cell_index);

    // Get cell dofmap
    auto cell_dofs0 = dofmap0.cell_dofs(cell.index());
    auto cell_dofs1 = dofmap1.cell_dofs(cell.index());

    // Loop over facet dofs
    const int entity_local_index = cell.index(entity);
    for (int i = 0; i < num_entity_dofs; ++i)
    {
      const int index = entity_dofs[entity_local_index][i];
      const PetscInt dof_index0 = cell_dofs0[index];
      const PetscInt dof_index1 = cell_dofs1[index];
      bc_dofs.push_back({{dof_index0, dof_index1}});
    }
  }

  // TODO: is removing duplicates at this point worth the effort?
  // Remove duplicates
  std::sort(bc_dofs.begin(), bc_dofs.end());
  bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());

  if (remote)
  {
    // Get bc dof indices (local) in (V, Vg) spaces on this process that
    // were found by other processes, e.g. a vertex dof on this process that
    // has no connected facets on the boundary.
    const std::vector<std::array<PetscInt, 2>> dofs_remote = get_remote_bcs(
        *V0.dofmap()->index_map, *V1.dofmap()->index_map, bc_dofs);

    // Add received bc indices to dofs_local
    for (auto& dof_remote : dofs_remote)
      bc_dofs.push_back(dof_remote);

    // TODO: is removing duplicates at this point worth the effort?
    // Remove duplicates
    std::sort(bc_dofs.begin(), bc_dofs.end());
    bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());
  }

  Eigen::Array<PetscInt, Eigen::Dynamic, 2> dofs(
      bc_dofs.size(), 2);
  for (std::size_t i = 0; i < bc_dofs.size(); ++i)
  {
    dofs(i, 0) = bc_dofs[i][0];
    dofs(i, 1) = bc_dofs[i][1];
  }

  return dofs;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 1> fem::locate_dofs_topological(
    const function::FunctionSpace& V, const int entity_dim,
    const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& entities,
    bool remote)
{
  assert(V.dofmap());
  const DofMap& dofmap = *V.dofmap();
  assert(V.mesh());
  dolfin::mesh::Mesh mesh = *V.mesh();

  const int tdim = mesh.topology().dim();

  // Initialise entity-cell connectivity
  mesh.create_entities(tdim);
  mesh.create_connectivity(entity_dim, tdim);

  // Prepare an element - local dof layout for dofs on entities of the
  // entity_dim
  const int num_cell_entities
      = mesh::cell_num_entities(mesh.cell_type(), entity_dim);
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> entity_dofs;
  for (int i = 0; i < num_cell_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap.element_dof_layout->entity_closure_dofs(entity_dim, i));
  }

  const int num_entity_closure_dofs
      = dofmap.element_dof_layout->num_entity_closure_dofs(entity_dim);
  std::vector<PetscInt> dofs;
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    // Create entity and attached cell
    const mesh::MeshEntity entity(mesh, entity_dim, entities[i]);
    const int cell_index = entity.entities(tdim)[0];
    const mesh::MeshEntity cell(mesh, tdim, cell_index);

    // Get cell dofmap
    auto cell_dofs = dofmap.cell_dofs(cell_index);

    // Loop over entity dofs
    const int entity_local_index = cell.index(entity);
    for (int j = 0; j < num_entity_closure_dofs; j++)
    {
      const int index = entity_dofs[entity_local_index][j];
      const PetscInt dof_index = cell_dofs[index];
      dofs.push_back(dof_index);
    }
  }

  // TODO: is removing duplicates at this point worth the effort?
  // Remove duplicates
  std::sort(dofs.begin(), dofs.end());
  dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());

  if (remote)
  {
    const std::vector<PetscInt> dofs_remote = get_remote_bcs(
        *V.dofmap()->index_map, dofs);

    // Add received bc indices to dofs_local
    for (auto& dof_remote : dofs_remote)
      dofs.push_back(dof_remote);

    // TODO: is removing duplicates at this point worth the effort?
    // Remove duplicates
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
  }

  return Eigen::Map<Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(dofs.data(),
                                                               dofs.size());
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 1>
fem::locate_dofs_geometrical(const function::FunctionSpace& V,
                             marking_function marker)
{
  // FIXME: Calling V.tabulate_dof_coordinates() is very expensive,
  // especially when we usually want the boundary dofs only. Add
  // interface that computes dofs coordinates only for specified cell.

  // Compute dof coordinates
  const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> dof_coordinates
      = V.tabulate_dof_coordinates().transpose();

  // Compute marker for each dof coordinate
  const Eigen::Array<bool, Eigen::Dynamic, 1> marked_dofs
      = marker(dof_coordinates);

  std::vector<PetscInt> dofs;
  dofs.reserve(marked_dofs.count());
  for (Eigen::Index i = 0; i < marked_dofs.size(); ++i)
  {
    if (marked_dofs[i])
      dofs.push_back(i);
  }

  return Eigen::Map<Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(dofs.data(),
                                                               dofs.size());
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(
    std::shared_ptr<const function::Function> g,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& V_dofs)
    : _function_space(g->function_space()), _g(g)
{

  _dofs = Eigen::Array<PetscInt, Eigen::Dynamic, 2>(V_dofs.rows(), 2);

  // Stack indices as columns, fits column-major _dofs layout
  _dofs.col(0) = V_dofs;
  _dofs.col(1) = V_dofs;

  const int owned_size = _function_space->dofmap()->index_map->block_size
                         * _function_space->dofmap()->index_map->size_local();
  auto it = std::lower_bound(_dofs.col(0).data(),
                             _dofs.col(0).data() + _dofs.rows(), owned_size);
  _owned_indices = std::distance(_dofs.col(0).data(), it);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(
    std::shared_ptr<const function::Function> g,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 2>>& V_g_dofs,
    std::shared_ptr<const function::FunctionSpace> V)
    : _function_space(V), _g(g), _dofs(V_g_dofs)
{
  const int owned_size = _function_space->dofmap()->index_map->block_size
                         * _function_space->dofmap()->index_map->size_local();
  auto it = std::lower_bound(_dofs.col(0).data(),
                             _dofs.col(0).data() + _dofs.rows(), owned_size);
  _owned_indices = std::distance(_dofs.col(0).data(), it);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::FunctionSpace>
DirichletBC::function_space() const
{
  return _function_space;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::Function> DirichletBC::value() const
{
  return _g;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 2>& DirichletBC::dofs() { return _dofs; }
//-----------------------------------------------------------------------------
const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 2>>
DirichletBC::dofs_owned() const
{
  return _dofs.block<Eigen::Dynamic, 2>(0, 0, _owned_indices, 2);
}
//-----------------------------------------------------------------------------
void DirichletBC::set(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
    double scale) const
{
  // FIXME: This one excludes ghosts. Need to straighten out.
  assert(_g);
  la::VecReadWrapper g(_g->vector().vec(), false);
  for (Eigen::Index i = 0; i < _dofs.rows(); ++i)
  {
    if (_dofs(i, 0) < x.rows())
      x[_dofs(i, 0)] = scale * g.x[_dofs(i, 1)];
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::set(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
    double scale) const
{
  // FIXME: This one excludes ghosts. Need to straighten out.
  assert(_g);
  assert(x.rows() <= x0.rows());
  la::VecReadWrapper g(_g->vector().vec(), false);
  for (Eigen::Index i = 0; i < _dofs.rows(); ++i)
  {
    if (_dofs(i, 0) < x.rows())
      x[_dofs(i, 0)] = scale * (g.x[_dofs(i, 1)] - x0[_dofs(i, 0)]);
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::dof_values(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> values) const
{
  assert(_g);
  la::VecReadWrapper g(_g->vector().vec());
  for (Eigen::Index i = 0; i < _dofs.rows(); ++i)
    values[_dofs(i, 0)] = g.x[_dofs(i, 1)];
}
//-----------------------------------------------------------------------------
void DirichletBC::mark_dofs(std::vector<bool>& markers) const
{
  for (Eigen::Index i = 0; i < _dofs.rows(); ++i)
  {
    assert(_dofs(i, 0) < (PetscInt)markers.size());
    markers[_dofs(i, 0)] = true;
  }
}
//-----------------------------------------------------------------------------
