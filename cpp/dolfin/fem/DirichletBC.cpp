// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DirichletBC.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include <array>
#include <dolfin/common/IndexMap.h>
#include <dolfin/fem/CoordinateElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/cell_types.h>
#include <map>
#include <utility>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
std::vector<std::array<PetscInt, 2>>
get_remote_bcs(const common::IndexMap& map, const common::IndexMap& map_g,
               const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs_local,
               const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs_local_g)
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
  for (Eigen::Index i = 0; i < dofs_local.rows(); i++)
  {
    const PetscInt index_block_g = dofs_local_g[i] / bs_g;
    const PetscInt pos_g = dofs_local_g[i] % bs_g;
    if (dofs_local[i] < bs * size_owned)
    {
      marker_owned[dofs_local[i]]
          = bs_g * map_g.local_to_global(index_block_g) + pos_g;
    }
    else
    {
      marker_ghost[dofs_local[i] - (bs * size_owned)]
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
DirichletBC::DirichletBC(
    std::shared_ptr<const function::FunctionSpace> V,
    std::shared_ptr<const function::Function> g,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& V_dofs,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& g_dofs)
    : _function_space(V), _g(g)
{

  if(V_dofs.rows() != g_dofs.rows())
  {
    throw std::runtime_error("Not matching number of degrees of freedom.");
  }

  // Get bc dof indices (local) in (V, Vg) spaces on this process that
  // were found by other processes, e.g. a vertex dof on this process that
  // has no connected facets on the boundary.
  const std::vector<std::array<PetscInt, 2>> dofs_remote
      = get_remote_bcs(*V->dofmap()->index_map,
                       *g->function_space()->dofmap()->index_map, V_dofs, g_dofs);

  // Copy the Eigen data structure into std::vector of arrays
  // This is needed for appending the remote dof indices and std::sort
  std::vector<std::array<PetscInt, 2>> dofs_local_vec(V_dofs.rows());
  for (Eigen::Index i = 0; i < V_dofs.rows(); ++i){
    dofs_local_vec[i] = {V_dofs[i], g_dofs[i]};
  }

  // Add received bc indices to dofs_local
  for (auto& dof_remote : dofs_remote)
    dofs_local_vec.push_back(dof_remote);

  // Remove duplicates
  std::sort(dofs_local_vec.begin(), dofs_local_vec.end());
  dofs_local_vec.erase(std::unique(dofs_local_vec.begin(), dofs_local_vec.end()),
                       dofs_local_vec.end());

  _dofs = Eigen::Array<PetscInt, Eigen::Dynamic, 2, Eigen::RowMajor>(
      dofs_local_vec.size(), 2);

  for (std::size_t i = 0; i < dofs_local_vec.size(); ++i)
  {
    _dofs(i, 0) = dofs_local_vec[i][0];
    _dofs(i, 1) = dofs_local_vec[i][1];
  }

  const int owned_size = V->dofmap()->index_map->block_size
                         * V->dofmap()->index_map->size_local();
  auto it
      = std::lower_bound(_dofs.col(0).data(),
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
Eigen::Array<PetscInt, Eigen::Dynamic, 2>&
DirichletBC::dofs()
{
  return _dofs;
}
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
  assert(_g);
  assert(x.rows() <= x0.rows());
  la::VecReadWrapper g(_g->vector().vec(), false);
  for (Eigen::Index i = 0; i < _dofs.rows(); ++i)
  {
    x[_dofs(i, 0)] = scale * (g.x[_dofs(i, 0)] - x0[_dofs(i, 0)]);
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
  g.restore();
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
Eigen::Array<PetscInt, Eigen::Dynamic, 1> fem::locate_dofs_topological(
    const function::FunctionSpace& V, const int entity_dim,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& entities)
{

  const DofMap& dofmap = *V.dofmap();
  dolfin::mesh::Mesh mesh = *V.mesh();

  const int tdim = mesh.topology().dim();

  // Initialise entity-cell connectivity
  mesh.create_entities(tdim);
  mesh.create_connectivity(entity_dim, tdim);

  // Prepare an element-local dof layout for dofs on entities of the entity_dim
  const int num_entities
      = mesh::cell_num_entities(mesh.cell_type(), entity_dim);
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> entity_dofs;
  for (int i = 0; i < num_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap.element_dof_layout->entity_closure_dofs(entity_dim, i));
  }

  const std::size_t num_entity_closure_dofs
      = dofmap.element_dof_layout->num_entity_closure_dofs(entity_dim);

  std::vector<PetscInt> dofs;

  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    // Create entity and attached cell
    const mesh::MeshEntity entity(mesh, entity_dim, entities[i]);
    const std::size_t cell_index = entity.entities(tdim)[0];
    const mesh::MeshEntity cell(mesh, tdim, cell_index);

    // Get cell dofmap
    auto cell_dofs = dofmap.cell_dofs(cell_index);

    // Loop over entity dofs
    const size_t entity_local_index = cell.index(entity);
    for (std::size_t j = 0; j < num_entity_closure_dofs; j++)
    {
      const std::size_t index = entity_dofs[entity_local_index][j];
      const PetscInt dof_index = cell_dofs[index];
      dofs.push_back(dof_index);
    }
  }

  return Eigen::Map<Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(dofs.data(),
                                                               dofs.size());
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 1>
fem::locate_dofs_geometrical(const function::FunctionSpace& V,
                             marking_function marker)
{

  const auto dof_coordinates = V.tabulate_dof_coordinates().transpose();
  const auto marked_dofs = marker(dof_coordinates);

  std::vector<PetscInt> dofs;

  for (PetscInt i = 0; i < marked_dofs.size(); ++i)
  {
    if (marked_dofs[i])
      dofs.push_back(i);
  }

  return Eigen::Map<Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(dofs.data(),
                                                               dofs.size());
}
