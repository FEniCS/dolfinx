// Copyright (C) 2007-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DirichletBC.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include <algorithm>
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
//-----------------------------------------------------------------------------

/// Find DOFs on this processes that are constrained by a Dirichlet
/// condition detected by another process
///
/// @param[in] map The IndexMap with the dof layout
/// @param[in] dofs_local The IndexMap with the dof layout
/// @return List of local dofs with boundary conditions applied but
///   detected by other processes. It may contain duplicate entries.
std::vector<std::int32_t>
get_remote_bcs_new(const common::IndexMap& map,
                   const std::vector<std::int32_t>& dofs_local)
{
  // Get number of processes in neighbourhood
  MPI_Comm comm = map.mpi_comm_neighborhood();
  int num_neighbours(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &num_neighbours, &outdegree, &weighted);
  assert(num_neighbours == outdegree);

  // Return early if there are no neighbours
  if (num_neighbours == 0)
    return std::vector<std::int32_t>();

  // Figure out how many entries to receive from each neighbour
  const int num_dofs = dofs_local.size();
  std::vector<int> num_dofs_recv(num_neighbours);
  MPI_Neighbor_allgather(&num_dofs, 1, MPI_INT, num_dofs_recv.data(), 1,
                         MPI_INT, comm);

  // Build array of global indices of dofs
  const int bs = map.block_size;
  std::vector<std::int64_t> dofs_global;
  dofs_global.reserve(dofs_local.size());
  for (auto dof : dofs_local)
  {
    const int index_block = dof / bs;
    const int pos = dof % bs;
    dofs_global.push_back(bs * map.local_to_global(index_block) + pos);
  }

  // Compute displacements for data to receive. Last entry has total
  // number of received items.
  std::vector<int> disp(num_neighbours + 1, 0);
  std::inclusive_scan(num_dofs_recv.begin(), num_dofs_recv.end(),
                      disp.begin() + 1);

  // Send/receive global index of dofs with bcs to all neighbours
  std::vector<std::int64_t> dofs_received(disp.back());
  MPI_Neighbor_allgatherv(dofs_global.data(), dofs_global.size(), MPI_INT64_T,
                          dofs_received.data(), num_dofs_recv.data(),
                          disp.data(), MPI_INT64_T, comm);

  // Build global-to-local map for ghost indices (blocks) on this
  // process
  const std::int32_t size_owned = map.size_local();
  std::map<std::int64_t, std::int32_t> global_to_local_blocked;
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghosts = map.ghosts();
  for (Eigen::Index i = 0; i < ghosts.rows(); ++i)
    global_to_local_blocked.insert({ghosts[i], i + size_owned});

  // Build vector of local dof indicies that have been marked by another
  // process
  std::vector<std::int32_t> dofs;
  const std::array<std::int64_t, 2> range = map.local_range();
  for (auto dof : dofs_received)
  {
    if (dof >= bs * range[0] and dof < bs * range[1])
      dofs.push_back(dof - bs * range[0]);
    else
    {
      const std::int64_t index_block = dof / bs;
      auto it = global_to_local_blocked.find(index_block);
      if (it != global_to_local_blocked.end())
        dofs.push_back(it->second * bs + dof % bs);
    }
  }

  return dofs;
}
//-----------------------------------------------------------------------------
// TODO: add some docs
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
//-----------------------------------------------------------------------------
// std::vector<PetscInt> get_remote_bcs(const common::IndexMap& map,
//                                      const std::vector<PetscInt>& dofs_local)
// {
//   const std::int32_t bs = map.block_size;
//   const std::int32_t size_owned = map.size_local();
//   const std::int32_t size_ghost = map.num_ghosts();

//   const std::array<std::int64_t, 2> range = map.local_range();
//   const std::int64_t offset = range[0];

//   // For each dof local index, store global index (-1 if no bc)
//   std::vector<PetscInt> marker_owned(bs * size_owned, -1);
//   std::vector<PetscInt> marker_ghost(bs * size_ghost, -1);
//   for (auto& dofs : dofs_local)
//   {
//     const PetscInt index_block = dofs / bs;
//     const PetscInt pos = dofs % bs;
//     if (dofs < bs * size_owned)
//       marker_owned[dofs] = bs * map.local_to_global(index_block) + pos;
//     else
//     {
//       marker_ghost[dofs - (bs * size_owned)]
//           = bs * map.local_to_global(index_block) + pos;
//     }
//   }

//   // Build global-to-local map for ghost indices (blocks) in map
//   std::map<PetscInt, PetscInt> global_to_local;
//   const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghosts = map.ghosts();
//   for (Eigen::Index i = 0; i < size_owned; ++i)
//     global_to_local.insert({i + offset, i});
//   for (Eigen::Index i = 0; i < size_ghost; ++i)
//     global_to_local.insert({ghosts[i], i + size_owned});

//   // For each owned bc index, scatter associated global index to ghost
//   // processes
//   std::vector<PetscInt> marker_ghost_rcvd = map.scatter_fwd(marker_owned,
//   bs); assert((int)marker_ghost_rcvd.size() == size_ghost * bs);

//   // Add to local indices map
//   std::vector<PetscInt> dofs;
//   for (std::size_t i = 0; i < marker_ghost_rcvd.size(); ++i)
//   {
//     if (marker_ghost_rcvd[i] > -1)
//     {
//       const PetscInt index_block = marker_ghost_rcvd[i] / bs;
//       const auto it = global_to_local.find(index_block);
//       assert(it != global_to_local.end());
//       dofs.push_back((PetscInt)(bs * size_owned + i));
//     }
//   }

//   return dofs;
// }
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 2> _locate_dofs_topological(
    const std::vector<std::reference_wrapper<function::FunctionSpace>>& V,
    const int dim, const Eigen::Ref<const Eigen::ArrayXi>& entities,
    bool remote)
{
  const function::FunctionSpace& V0 = V.at(0).get();
  const function::FunctionSpace& V1 = V.at(1).get();

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
      bc_dofs.push_back({cell_dofs0[index], cell_dofs1[index]});
    }
  }

  // TODO: is removing duplicates at this point worth the effort?
  // Remove duplicates
  std::sort(bc_dofs.begin(), bc_dofs.end());
  bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());

  if (remote)
  {
    // Get bc dof indices (local) in (V, Vg) spaces on this process that
    // were found by other processes, e.g. a vertex dof on this process
    // that has no connected facets on the boundary.
    const std::vector<std::array<PetscInt, 2>> dofs_remote = get_remote_bcs(
        *V0.dofmap()->index_map, *V1.dofmap()->index_map, bc_dofs);

    // Add received bc indices to dofs_local
    bc_dofs.insert(bc_dofs.end(), dofs_remote.begin(), dofs_remote.end());

    // TODO: is removing duplicates at this point worth the effort?
    // Remove duplicates
    std::sort(bc_dofs.begin(), bc_dofs.end());
    bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());
  }

  Eigen::Array<PetscInt, Eigen::Dynamic, 2> dofs(bc_dofs.size(), 2);
  for (std::size_t i = 0; i < bc_dofs.size(); ++i)
  {
    dofs(i, 0) = bc_dofs[i][0];
    dofs(i, 1) = bc_dofs[i][1];
  }

  return dofs;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 1>
_locate_dofs_topological(const function::FunctionSpace& V, const int entity_dim,
                         const Eigen::Ref<const Eigen::ArrayXi>& entities,
                         bool remote)
{
  assert(V.dofmap());
  const DofMap& dofmap = *V.dofmap();
  assert(V.mesh());
  mesh::Mesh mesh = *V.mesh();

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
  std::vector<std::int32_t> dofs;
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
      dofs.push_back(cell_dofs[index]);
    }
  }

  // TODO: is removing duplicates at this point worth the effort?
  // Remove duplicates
  std::sort(dofs.begin(), dofs.end());
  dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());

  if (remote)
  {
    const std::vector<std::int32_t> dofs_remote
        = get_remote_bcs_new(*V.dofmap()->index_map, dofs);

    // Add received bc indices to dofs_local
    dofs.insert(dofs.end(), dofs_remote.begin(), dofs_remote.end());

    // Remove duplicates
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
  }

  // Copy to array of PetscInt type
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _dofs
      = Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(dofs.data(),
                                                                  dofs.size());

  return _dofs;
}
} // namespace
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
fem::locate_dofs_topological(
    const std::vector<std::reference_wrapper<function::FunctionSpace>>& V,
    const int dim, const Eigen::Ref<const Eigen::ArrayXi>& entities,
    bool remote)
{
  if (V.size() == 2)
    return _locate_dofs_topological(V, dim, entities, remote);
  else if (V.size() == 1)
    return _locate_dofs_topological(V[0].get(), dim, entities, remote);
  else
    throw std::runtime_error("Expected only 1 or 2 function spaces.");
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
  for (Eigen::Index i = 0; i < marked_dofs.rows(); ++i)
  {
    if (marked_dofs[i])
      dofs.push_back(i);
  }

  return Eigen::Map<Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(dofs.data(),
                                                               dofs.size());
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(
    std::shared_ptr<const function::Function> g,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& V_dofs)
    : _function_space(g->function_space()), _g(g), _dofs(V_dofs.rows(), 2)
{
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
