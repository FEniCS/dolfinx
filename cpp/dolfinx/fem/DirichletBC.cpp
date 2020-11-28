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
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <map>
#include <numeric>
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
get_remote_bcs1(const common::IndexMap& map, int bs,
                const std::vector<std::int32_t>& dofs_local)
{
  MPI_Comm comm = map.comm(common::IndexMap::Direction::symmetric);

  // Get number of processes in neighborhood
  int num_neighbors(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &num_neighbors, &outdegree, &weighted);

  // Return early if there are no neighbors
  if (num_neighbors == 0)
    return {};

  // Figure out how many entries to receive from each neighbor
  const int num_dofs = dofs_local.size();
  std::vector<int> num_dofs_recv(num_neighbors);
  MPI_Neighbor_allgather(&num_dofs, 1, MPI_INT, num_dofs_recv.data(), 1,
                         MPI_INT, comm);

  // NOTE: we could consider only dofs that we know are shared
  // Build array of global indices of dofs
  std::vector<std::int32_t> dofs_local_block = dofs_local;
  std::for_each(dofs_local_block.begin(), dofs_local_block.end(),
                [bs](std::int32_t& n) { n /= bs; });
  const std::vector<std::int64_t> dofs_global_block
      = map.local_to_global(dofs_local_block);
  std::vector<std::int64_t> dofs_global(dofs_local.size());
  for (std::size_t i = 0; i < dofs_local.size(); ++i)
  {
    const int offset = dofs_local[i] % bs;
    dofs_global[i] = bs * dofs_global_block[i] + offset;
  }

  // Compute displacements for data to receive. Last entry has total
  // number of received items.
  // Note: std::inclusive_scan would be better, but requires gcc >= 9.2.1
  std::vector<int> disp(num_neighbors + 1, 0);
  std::partial_sum(num_dofs_recv.begin(), num_dofs_recv.end(),
                   disp.begin() + 1);
  // std::inclusive_scan(num_dofs_recv.begin(), num_dofs_recv.end(),
  //                     disp.begin() + 1);

  // NOTE: we could use MPI_Neighbor_alltoallv to send only to relevant
  // processes

  // Send/receive global index of dofs with bcs to all neighbors
  std::vector<std::int64_t> dofs_received(disp.back());
  MPI_Neighbor_allgatherv(dofs_global.data(), dofs_global.size(), MPI_INT64_T,
                          dofs_received.data(), num_dofs_recv.data(),
                          disp.data(), MPI_INT64_T, comm);

  // FIXME: check that dofs is sorted
  // Build vector of local dof indicies that have been marked by another
  // process
  const std::array<std::int64_t, 2> range = map.local_range();
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts = map.ghosts();

  // Build map from ghost to local position
  std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts;
  const std::int32_t local_size = range[1] - range[0];
  for (Eigen::Index i = 0; i < ghosts.rows(); ++i)
    global_local_ghosts.emplace_back(ghosts[i], i + local_size);
  std::map<std::int64_t, std::int32_t> global_to_local(
      global_local_ghosts.begin(), global_local_ghosts.end());

  std::vector<std::int32_t> dofs;
  for (std::size_t i = 0; i < dofs_received.size(); ++i)
  {
    if (dofs_received[i] >= bs * range[0] and dofs_received[i] < bs * range[1])
      dofs.push_back(dofs_received[i] - bs * range[0]);
    else
    {
      // Search in ghosts
      if (auto it = global_to_local.find(dofs_received[i] / bs);
          it != global_to_local.end())
      {
        dofs.push_back(it->second * bs + dofs_received[i] % bs);
      }
    }
  }

  return dofs;
}
//-----------------------------------------------------------------------------

/// Find DOFs on this processes that are constrained by a Dirichlet
/// condition detected by another process
///
/// @param[in] map0 The IndexMap with the dof layout
/// @param[in] map1 The IndexMap with the dof layout
/// @param[in] dofs_local The IndexMap with the dof layout
/// @return List of local dofs with boundary conditions applied but
///   detected by other processes. It may contain duplicate entries.
std::vector<std::array<std::int32_t, 2>>
get_remote_bcs2(const common::IndexMap& map0, int bs0,
                const common::IndexMap& map1, int bs1,
                const std::vector<std::array<std::int32_t, 2>>& dofs_local)
{
  MPI_Comm comm0 = map0.comm(common::IndexMap::Direction::symmetric);

  int num_neighbors(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm0, &num_neighbors, &outdegree, &weighted);
  assert(num_neighbors == outdegree);

  // Return early if there are no neighbors
  if (num_neighbors == 0)
    return {};

  // Figure out how many entries to receive from each neighbor
  const int num_dofs = 2 * dofs_local.size();
  std::vector<int> num_dofs_recv(num_neighbors);
  MPI_Neighbor_allgather(&num_dofs, 1, MPI_INT, num_dofs_recv.data(), 1,
                         MPI_INT, comm0);

  // NOTE: we consider only dofs that we know are shared
  // Build array of global indices of dofs
  Eigen::Array<std::int64_t, Eigen::Dynamic, 2, Eigen::RowMajor> dofs_global(
      dofs_local.size(), 2);

  // This is messy to handle block sizes
  {
    const std::array<int, 2> _bs = {bs0, bs1};
    const std::array<std::reference_wrapper<const common::IndexMap>, 2> maps
        = {map0, map1};
    std::vector<std::int32_t> dofs_local_block, _dofs_local(dofs_local.size());
    std::vector<std::int64_t> dofs_global_block;
    for (int i = 0; i < 2; ++i)
    {
      for (std::size_t j = 0; j < _dofs_local.size(); ++j)
        _dofs_local[j] = dofs_local[j][i];

      dofs_local_block = _dofs_local;
      std::for_each(dofs_local_block.begin(), dofs_local_block.end(),
                    [bs = _bs[i]](std::int32_t& n) { return n /= bs; });
      dofs_global_block = maps[i].get().local_to_global(dofs_local_block);
      for (std::size_t j = 0; j < dofs_local.size(); ++j)
      {
        const int offset = _dofs_local[j] % _bs[i];
        dofs_global(j, i) = _bs[i] * dofs_global_block[j] + offset;
      }
    }
  }

  // Compute displacements for data to receive. Last entry has total
  // number of received items.
  // Note: std::inclusive_scan would be better, but requires gcc >= 9.2.1
  std::vector<int> disp(num_neighbors + 1, 0);
  std::partial_sum(num_dofs_recv.begin(), num_dofs_recv.end(),
                   disp.begin() + 1);
  // std::inclusive_scan(num_dofs_recv.begin(), num_dofs_recv.end(),
  //                     disp.begin() + 1);

  // NOTE: we could use MPI_Neighbor_alltoallv to send only to relevant
  // processes

  // Send/receive global index of dofs with bcs to all neighbors
  // std::vector<std::int64_t> dofs_received(disp.back());
  assert(disp.back() % 2 == 0);
  Eigen::Array<std::int64_t, Eigen::Dynamic, 2, Eigen::RowMajor> dofs_received(
      disp.back() / 2, 2);
  MPI_Neighbor_allgatherv(dofs_global.data(), dofs_global.size(), MPI_INT64_T,
                          dofs_received.data(), num_dofs_recv.data(),
                          disp.data(), MPI_INT64_T, comm0);

  const std::array<std::reference_wrapper<const common::IndexMap>, 2> maps
      = {map0, map1};
  const std::array bs = {bs0, bs1};
  std::array<std::vector<std::int32_t>, 2> dofs_array;
  for (int b = 0; b < 2; ++b)
  {
    // FIXME: check that dofs is sorted
    // Build vector of local dof indicies that have been marked by another
    // process
    const std::array<std::int64_t, 2> range = maps[b].get().local_range();
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts
        = maps[b].get().ghosts();

    // Build map from ghost to local position
    std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts;
    const std::int32_t local_size = range[1] - range[0];
    for (Eigen::Index i = 0; i < ghosts.rows(); ++i)
      global_local_ghosts.emplace_back(ghosts[i], i + local_size);
    std::map<std::int64_t, std::int32_t> global_to_local(
        global_local_ghosts.begin(), global_local_ghosts.end());
    std::vector<std::int32_t>& dofs = dofs_array[b];
    for (Eigen::Index i = 0; i < dofs_received.rows(); ++i)
    {
      if (dofs_received(i, b) >= bs[b] * range[0]
          and dofs_received(i, b) < bs[b] * range[1])
      {
        // Owned dof
        dofs.push_back(dofs_received(i, b) - bs[b] * range[0]);
      }
      else
      {
        // Search in ghosts
        if (auto it = global_to_local.find(dofs_received(i, b) / bs[b]);
            it != global_to_local.end())
        {
          dofs.push_back(it->second * bs[b] + dofs_received(i, b) % bs[b]);
        }
      }
    }
  }
  assert(dofs_array[0].size() == dofs_array[1].size());

  std::vector<std::array<std::int32_t, 2>> dofs;
  dofs.reserve(dofs_array[0].size());
  for (std::size_t i = 0; i < dofs_array[0].size(); ++i)
    dofs.push_back({dofs_array[0][i], dofs_array[1][i]});

  return dofs;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>
fem::locate_dofs_topological(
    const std::array<std::reference_wrapper<const function::FunctionSpace>, 2>&
        V,
    const int dim, const Eigen::Ref<const Eigen::ArrayXi>& entities,
    bool remote)
{
  const function::FunctionSpace& V0 = V.at(0).get();
  const function::FunctionSpace& V1 = V.at(1).get();

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::runtime_error("Meshes are not the same.");
  const int tdim = mesh->topology().dim();

  assert(V0.element());
  assert(V1.element());
  if (!V0.has_element(*V1.element()))
  {
    throw std::runtime_error("Function spaces must have the same elements or "
                             "one be a subelement of another.");
  }

  // Get dofmaps
  std::shared_ptr<const fem::DofMap> dofmap0 = V0.dofmap();
  std::shared_ptr<const fem::DofMap> dofmap1 = V1.dofmap();
  assert(dofmap0);
  assert(dofmap1);

  // Initialise entity-cell connectivity
  // FIXME: cleanup these calls? Some of the happen internally again.
  mesh->topology_mutable().create_entities(tdim);
  mesh->topology_mutable().create_connectivity(dim, tdim);

  // Allocate space
  assert(dofmap0->element_dof_layout);
  const int num_entity_dofs
      = dofmap0->element_dof_layout->num_entity_closure_dofs(dim);
  const int block_size = dofmap0->element_dof_layout->block_size();
  assert(block_size == dofmap1->element_dof_layout->block_size());

  // Build vector local dofs for each cell facet
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> entity_dofs;
  for (int i = 0;
       i < mesh::cell_num_entities(mesh->topology().cell_type(), dim); ++i)
  {
    entity_dofs.push_back(
        dofmap0->element_dof_layout->entity_closure_dofs(dim, i));
  }
  auto e_to_c = mesh->topology().connectivity(dim, tdim);
  assert(e_to_c);
  auto c_to_e = mesh->topology().connectivity(tdim, dim);
  assert(c_to_e);

  // Iterate over marked facets
  std::vector<std::array<std::int32_t, 2>> bc_dofs;
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    // Get first attached cell
    assert(e_to_c->num_links(entities[e]) > 0);
    const int cell = e_to_c->links(entities[e])[0];

    // Get local index of facet with respect to the cell
    auto entities_d = c_to_e->links(cell);
    const auto* it = std::find(
        entities_d.data(), entities_d.data() + entities_d.rows(), entities[e]);
    assert(it != (entities_d.data() + entities_d.rows()));
    const int entity_local_index = std::distance(entities_d.data(), it);

    // Get cell dofmap
    auto cell_dofs0 = dofmap0->cell_dofs(cell);
    auto cell_dofs1 = dofmap1->cell_dofs(cell);

    // Loop over facet dofs
    for (int i = 0; i < num_entity_dofs; ++i)
    {
      const int index = entity_dofs[entity_local_index][i];
      for (int block = 0; block < block_size; ++block)
      {
        bc_dofs.push_back({cell_dofs0[index * block_size + block],
                           cell_dofs1[index * block_size + block]});
      }
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
    const std::vector dofs_remote = get_remote_bcs2(
        *V0.dofmap()->index_map, V0.dofmap()->index_map_bs(),
        *V1.dofmap()->index_map, V1.dofmap()->index_map_bs(), bc_dofs);

    // Add received bc indices to dofs_local
    bc_dofs.insert(bc_dofs.end(), dofs_remote.begin(), dofs_remote.end());

    // Remove duplicates
    std::sort(bc_dofs.begin(), bc_dofs.end());
    bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());
  }

  // Copy to Eigen array
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs0(bc_dofs.size());
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs1(bc_dofs.size());
  for (std::size_t i = 0; i < bc_dofs.size(); ++i)
  {
    dofs0(i) = bc_dofs[i][0];
    dofs1(i) = bc_dofs[i][1];
  }

  return {std::move(dofs0), std::move(dofs1)};
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
fem::locate_dofs_topological(const function::FunctionSpace& V, const int dim,
                             const Eigen::Ref<const Eigen::ArrayXi>& entities,
                             bool remote)
{
  assert(V.dofmap());
  std::shared_ptr<const DofMap> dofmap = V.dofmap();
  assert(V.mesh());
  std::shared_ptr<const mesh::Mesh> mesh = V.mesh();

  const int tdim = mesh->topology().dim();

  // Initialise entity-cell connectivity
  // FIXME: cleanup these calls? Some of them happen internally again.
  mesh->topology_mutable().create_entities(tdim);
  mesh->topology_mutable().create_connectivity(dim, tdim);

  // Prepare an element - local dof layout for dofs on entities of the
  // entity_dim
  const int num_cell_entities
      = mesh::cell_num_entities(mesh->topology().cell_type(), dim);
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> entity_dofs;
  for (int i = 0; i < num_cell_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap->element_dof_layout->entity_closure_dofs(dim, i));
  }

  auto e_to_c = mesh->topology().connectivity(dim, tdim);
  assert(e_to_c);
  auto c_to_e = mesh->topology().connectivity(tdim, dim);
  assert(c_to_e);

  const int num_entity_closure_dofs
      = dofmap->element_dof_layout->num_entity_closure_dofs(dim);
  const int block_size = dofmap->element_dof_layout->block_size();
  std::vector<std::int32_t> dofs;
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    // Get first attached cell
    assert(e_to_c->num_links(entities[i]) > 0);
    const int cell = e_to_c->links(entities[i])[0];

    // Get local index of facet with respect to the cell
    auto entities_d = c_to_e->links(cell);
    const auto* it = std::find(
        entities_d.data(), entities_d.data() + entities_d.rows(), entities[i]);
    assert(it != (entities_d.data() + entities_d.rows()));
    const int entity_local_index = std::distance(entities_d.data(), it);

    // Get cell dofmap
    auto cell_dofs = dofmap->cell_dofs(cell);

    // Loop over entity dofs
    for (int j = 0; j < num_entity_closure_dofs; j++)
    {
      const int index = entity_dofs[entity_local_index][j];
      for (int block = 0; block < block_size; ++block)
        dofs.push_back(cell_dofs[index * block_size + block]);
    }
  }

  // TODO: is removing duplicates at this point worth the effort?
  // Remove duplicates
  std::sort(dofs.begin(), dofs.end());
  dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());

  if (remote)
  {
    const std::vector dofs_remote = get_remote_bcs1(
        *V.dofmap()->index_map, V.dofmap()->index_map_bs(), dofs);

    // Add received bc indices to dofs_local
    dofs.insert(dofs.end(), dofs_remote.begin(), dofs_remote.end());

    // Remove duplicates
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
  }

  // Copy to Eigen array and return
  return Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(dofs.data(),
                                                                   dofs.size());
}
//-----------------------------------------------------------------------------
std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>
fem::locate_dofs_geometrical(
    const std::array<std::reference_wrapper<const function::FunctionSpace>, 2>&
        V,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker)
{
  // FIXME: Calling V.tabulate_dof_coordinates() is very expensive,
  // especially when we usually want the boundary dofs only. Add
  // interface that computes dofs coordinates only for specified cell.

  // Get function spaces
  const function::FunctionSpace& V0 = V.at(0).get();
  const function::FunctionSpace& V1 = V.at(1).get();

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::runtime_error("Meshes are not the same.");
  const int tdim = mesh->topology().dim();

  assert(V0.element());
  assert(V1.element());
  if (!V0.has_element(*V1.element()))
  {
    throw std::runtime_error("Function spaces must have the same elements or "
                             "one be a subelement of another.");
  }

  // Compute dof coordinates
  const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> dof_coordinates
      = V1.tabulate_dof_coordinates().transpose();

  // Evaluate marker for each dof coordinate
  const Eigen::Array<bool, Eigen::Dynamic, 1> marked_dofs
      = marker(dof_coordinates);

  // Get dofmaps
  std::shared_ptr<const fem::DofMap> dofmap0 = V0.dofmap();
  std::shared_ptr<const fem::DofMap> dofmap1 = V1.dofmap();
  assert(dofmap0);
  assert(dofmap1);

  // Iterate over cells
  const mesh::Topology& topology = mesh->topology();
  std::vector<std::array<std::int32_t, 2>> bc_dofs;
  for (int c = 0; c < topology.connectivity(tdim, 0)->num_nodes(); ++c)
  {
    // Get cell dofmap
    auto cell_dofs0 = dofmap0->cell_dofs(c);
    auto cell_dofs1 = dofmap1->cell_dofs(c);

    // Loop over cell dofs and add to bc_dofs if marked.
    for (Eigen::Index i = 0; i < cell_dofs1.rows(); ++i)
    {
      if (marked_dofs[cell_dofs1[i]])
      {
        bc_dofs.push_back(
            {(std::int32_t)cell_dofs0[i], (std::int32_t)cell_dofs1[i]});
      }
    }
  }

  // Remove duplicates
  std::sort(bc_dofs.begin(), bc_dofs.end());
  bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());

  // Copy to Eigen arrays
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs0(bc_dofs.size());
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs1(bc_dofs.size());
  for (std::size_t i = 0; i < bc_dofs.size(); ++i)
  {
    dofs0(i) = bc_dofs[i][0];
    dofs1(i) = bc_dofs[i][1];
  }

  return {std::move(dofs0), std::move(dofs1)};
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> fem::locate_dofs_geometrical(
    const function::FunctionSpace& V,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker)
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

  std::vector<std::int32_t> dofs;
  dofs.reserve(marked_dofs.count());
  for (Eigen::Index i = 0; i < marked_dofs.rows(); ++i)
  {
    if (marked_dofs[i])
      dofs.push_back(i);
  }

  return Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(dofs.data(),
                                                                   dofs.size());
}
//-----------------------------------------------------------------------------
