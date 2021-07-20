// Copyright (C) 2007-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DirichletBC.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include <algorithm>
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <map>
#include <numeric>
#include <utility>
#include <xtensor/xtensor.hpp>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
/// Create a symmetric MPI neighbourhood communciator from an
/// input neighbourhood communicator
dolfinx::MPI::Comm create_symmetric_comm(MPI_Comm comm)
{
  // assert(comm != MPI_COMM_NULL);
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);

  std::vector<int> neighbors(indegree + outdegree);
  MPI_Dist_graph_neighbors(comm, indegree, neighbors.data(), MPI_UNWEIGHTED,
                           outdegree, neighbors.data() + indegree,
                           MPI_UNWEIGHTED);

  std::sort(neighbors.begin(), neighbors.end());
  neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                  neighbors.end());

  MPI_Comm comm_sym;
  std::vector<int> sourceweights(neighbors.size(), 1);
  std::vector<int> destweights(neighbors.size(), 1);
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 sourceweights.data(), neighbors.size(),
                                 neighbors.data(), destweights.data(),
                                 MPI_INFO_NULL, false, &comm_sym);

  return dolfinx::MPI::Comm(comm_sym, false);
}

//-----------------------------------------------------------------------------
/// Find DOFs on this processes that are constrained by a Dirichlet
/// condition detected by another process
///
/// @param[in] map The IndexMap with the dof layout
/// @param[in] dofs_local The IndexMap with the dof layout
/// @return List of local dofs with boundary conditions applied but
///   detected by other processes. It may contain duplicate entries.
std::vector<std::int32_t>
get_remote_bcs1(const common::IndexMap& map,
                const std::vector<std::int32_t>& dofs_local)
{
  dolfinx::MPI::Comm comm
      = create_symmetric_comm(map.comm(common::IndexMap::Direction::forward));

  // Get number of processes in neighborhood
  int num_neighbors(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm.comm(), &num_neighbors, &outdegree,
                                 &weighted);

  // Return early if there are no neighbors
  if (num_neighbors == 0)
    return {};

  // Figure out how many entries to receive from each neighbor
  const int num_dofs = dofs_local.size();
  std::vector<int> num_dofs_recv(num_neighbors);
  MPI_Neighbor_allgather(&num_dofs, 1, MPI_INT, num_dofs_recv.data(), 1,
                         MPI_INT, comm.comm());

  // NOTE: we could consider only dofs that we know are shared
  // Build array of global indices of dofs
  std::vector<std::int64_t> dofs_global(dofs_local.size());
  map.local_to_global(dofs_local, dofs_global);

  // Compute displacements for data to receive. Last entry has total
  // number of received items.
  std::vector<int> disp(num_neighbors + 1, 0);
  std::partial_sum(num_dofs_recv.begin(), num_dofs_recv.end(),
                   std::next(disp.begin()));

  // NOTE: we could use MPI_Neighbor_alltoallv to send only to relevant
  // processes

  // Send/receive global index of dofs with bcs to all neighbors
  std::vector<std::int64_t> dofs_received(disp.back());
  MPI_Neighbor_allgatherv(dofs_global.data(), dofs_global.size(), MPI_INT64_T,
                          dofs_received.data(), num_dofs_recv.data(),
                          disp.data(), MPI_INT64_T, comm.comm());

  // FIXME: check that dofs is sorted
  // Build vector of local dof indicies that have been marked by another
  // process
  const std::array<std::int64_t, 2> range = map.local_range();
  const std::vector<std::int64_t>& ghosts = map.ghosts();

  // Build map from ghost to local position
  std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts;
  const std::int32_t local_size = range[1] - range[0];
  for (std::size_t i = 0; i < ghosts.size(); ++i)
    global_local_ghosts.emplace_back(ghosts[i], i + local_size);
  std::map<std::int64_t, std::int32_t> global_to_local(
      global_local_ghosts.begin(), global_local_ghosts.end());

  std::vector<std::int32_t> dofs;
  for (std::size_t i = 0; i < dofs_received.size(); ++i)
  {
    if (dofs_received[i] >= range[0] and dofs_received[i] < range[1])
      dofs.push_back(dofs_received[i] - range[0]);
    else
    {
      // Search in ghosts
      if (auto it = global_to_local.find(dofs_received[i]);
          it != global_to_local.end())
      {
        dofs.push_back(it->second);
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
  // NOTE: assumes that dofs are unrolled, i.e. not blocked. Could it be
  // make more efficient to handle the case of a common block size?

  dolfinx::MPI::Comm comm0
      = create_symmetric_comm(map0.comm(common::IndexMap::Direction::forward));

  int num_neighbors(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm0.comm(), &num_neighbors, &outdegree,
                                 &weighted);
  assert(num_neighbors == outdegree);

  // Return early if there are no neighbors
  if (num_neighbors == 0)
    return {};

  // Figure out how many entries to receive from each neighbor
  const int num_dofs = 2 * dofs_local.size();
  std::vector<int> num_dofs_recv(num_neighbors);
  MPI_Neighbor_allgather(&num_dofs, 1, MPI_INT, num_dofs_recv.data(), 1,
                         MPI_INT, comm0.comm());

  // NOTE: we consider only dofs that we know are shared
  // Build array of global indices of dofs
  xt::xtensor<std::int64_t, 2> dofs_global({dofs_local.size(), 2});

  // This is messy to handle block sizes
  {
    const std::array<int, 2> _bs = {bs0, bs1};
    const std::array<std::reference_wrapper<const common::IndexMap>, 2> maps
        = {map0, map1};
    std::vector<std::int32_t> _dofs_local(dofs_local.size());
    for (int i = 0; i < 2; ++i)
    {
      for (std::size_t j = 0; j < _dofs_local.size(); ++j)
        _dofs_local[j] = dofs_local[j][i];

      // Convert dofs indices to 'blocks' relative to index map
      std::vector<std::int32_t> dofs_local_block = _dofs_local;
      std::for_each(dofs_local_block.begin(), dofs_local_block.end(),
                    [bs = _bs[i]](std::int32_t& n) { return n /= bs; });

      // Get global index of each block
      std::vector<std::int64_t> dofs_global_block(dofs_local_block.size());
      maps[i].get().local_to_global(dofs_local_block, dofs_global_block);

      // Convert from block to actual index
      for (std::size_t j = 0; j < dofs_local.size(); ++j)
      {
        const int index_offset = _dofs_local[j] % _bs[i];
        dofs_global(j, i) = _bs[i] * dofs_global_block[j] + index_offset;
      }
    }
  }

  // Compute displacements for data to receive. Last entry has total
  // number of received items.
  std::vector<int> disp(num_neighbors + 1, 0);
  std::partial_sum(num_dofs_recv.begin(), num_dofs_recv.end(),
                   std::next(disp.begin()));

  // NOTE: we could use MPI_Neighbor_alltoallv to send only to relevant
  // processes

  // Send/receive global index of dofs with bcs to all neighbors
  assert(disp.back() % 2 == 0);
  xt::xtensor<std::int64_t, 2> dofs_received(
      {static_cast<std::size_t>(disp.back() / 2), 2});
  MPI_Neighbor_allgatherv(dofs_global.data(), dofs_global.size(), MPI_INT64_T,
                          dofs_received.data(), num_dofs_recv.data(),
                          disp.data(), MPI_INT64_T, comm0.comm());

  const std::array<std::reference_wrapper<const common::IndexMap>, 2> maps
      = {map0, map1};
  const std::array bs = {bs0, bs1};
  std::array<std::vector<std::int32_t>, 2> dofs_array;
  for (int b = 0; b < 2; ++b)
  {
    // FIXME: check that dofs is sorted?
    // Build vector of local dof indicies that have been marked by
    // another process
    const std::array<std::int64_t, 2> range = maps[b].get().local_range();
    const std::vector<std::int64_t>& ghosts = maps[b].get().ghosts();

    // Build map from ghost to local position
    std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts;
    const std::int32_t local_size = range[1] - range[0];
    for (std::size_t i = 0; i < ghosts.size(); ++i)
      global_local_ghosts.emplace_back(ghosts[i], i + local_size);
    std::map<std::int64_t, std::int32_t> global_to_local(
        global_local_ghosts.begin(), global_local_ghosts.end());

    std::vector<std::int32_t>& dofs = dofs_array[b];
    for (std::size_t i = 0; i < dofs_received.shape(0); ++i)
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
          dofs.push_back(bs[b] * it->second + dofs_received(i, b) % bs[b]);
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
std::array<std::vector<std::int32_t>, 2> fem::locate_dofs_topological(
    const std::array<std::reference_wrapper<const fem::FunctionSpace>, 2>& V,
    const int dim, const xtl::span<const std::int32_t>& entities, bool remote)
{
  const fem::FunctionSpace& V0 = V.at(0).get();
  const fem::FunctionSpace& V1 = V.at(1).get();

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::runtime_error("Meshes are not the same.");
  const int tdim = mesh->topology().dim();

  // FIXME: Elements must be the same?
  assert(V0.element());
  assert(V1.element());
  if (V0.element()->hash() != V1.element()->hash())
    throw std::runtime_error("Function spaces must have the same element.");

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
  // FIXME: check that dof layouts are the same
  assert(dofmap0->element_dof_layout);
  const int num_entity_dofs
      = dofmap0->element_dof_layout->num_entity_closure_dofs(dim);
  const int element_bs = dofmap0->element_dof_layout->block_size();
  assert(element_bs == dofmap1->element_dof_layout->block_size());

  // Build vector local dofs for each cell facet
  std::vector<std::vector<int>> entity_dofs;
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

  const int bs0 = dofmap0->bs();
  const int bs1 = dofmap1->bs();

  // Iterate over marked facets
  std::vector<std::array<std::int32_t, 2>> bc_dofs;
  for (std::int32_t e : entities)
  {
    // Get first attached cell
    assert(e_to_c->num_links(e) > 0);
    const int cell = e_to_c->links(e)[0];

    // Get local index of facet with respect to the cell
    auto entities_d = c_to_e->links(cell);
    auto it = std::find(entities_d.begin(), entities_d.end(), e);
    assert(it != entities_d.end());
    const int entity_local_index = std::distance(entities_d.begin(), it);

    // Get cell dofmap
    xtl::span<const std::int32_t> cell_dofs0 = dofmap0->cell_dofs(cell);
    xtl::span<const std::int32_t> cell_dofs1 = dofmap1->cell_dofs(cell);
    assert(bs0 * cell_dofs0.size() == bs1 * cell_dofs1.size());

    // Loop over facet dofs and 'unpack' blocked dofs
    for (int i = 0; i < num_entity_dofs; ++i)
    {
      const int index = entity_dofs[entity_local_index][i];
      for (int block = 0; block < element_bs; ++block)
      {
        const int local_pos = element_bs * index + block;
        const std::div_t pos0 = std::div(local_pos, bs0);
        const std::div_t pos1 = std::div(local_pos, bs1);
        const std::int32_t dof_index0 = bs0 * cell_dofs0[pos0.quot] + pos0.rem;
        const std::int32_t dof_index1 = bs1 * cell_dofs1[pos1.quot] + pos1.rem;
        bc_dofs.push_back({dof_index0, dof_index1});
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

    // Remove duplicates and sort
    std::sort(bc_dofs.begin(), bc_dofs.end());
    bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());
  }

  // Copy to separate vector
  std::array dofs = {std::vector<std::int32_t>(bc_dofs.size()),
                     std::vector<std::int32_t>(bc_dofs.size())};
  for (std::size_t i = 0; i < dofs[0].size(); ++i)
  {
    dofs[0][i] = bc_dofs[i][0];
    dofs[1][i] = bc_dofs[i][1];
  }

  return dofs;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
fem::locate_dofs_topological(const fem::FunctionSpace& V, const int dim,
                             const xtl::span<const std::int32_t>& entities,
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
  std::vector<std::vector<int>> entity_dofs;
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
  std::vector<std::int32_t> dofs;
  for (std::int32_t e : entities)
  {
    // Get first attached cell
    assert(e_to_c->num_links(e) > 0);
    const int cell = e_to_c->links(e)[0];

    // Get local index of facet with respect to the cell
    auto entities_d = c_to_e->links(cell);
    auto it = std::find(entities_d.begin(), entities_d.end(), e);
    assert(it != entities_d.end());
    const int entity_local_index = std::distance(entities_d.data(), it);

    // Get cell dofmap
    auto cell_dofs = dofmap->cell_dofs(cell);

    // Loop over entity dofs
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
    const std::vector dofs_remote
        = get_remote_bcs1(*V.dofmap()->index_map, dofs);

    // Add received bc indices to dofs_local
    dofs.insert(dofs.end(), dofs_remote.begin(), dofs_remote.end());

    // Remove duplicates
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
  }

  return dofs;
}
//-----------------------------------------------------------------------------
std::array<std::vector<std::int32_t>, 2> fem::locate_dofs_geometrical(
    const std::array<std::reference_wrapper<const fem::FunctionSpace>, 2>& V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker_fn)
{
  // FIXME: Calling V.tabulate_dof_coordinates() is very expensive,
  // especially when we usually want the boundary dofs only. Add
  // interface that computes dofs coordinates only for specified cell.

  // Get function spaces
  const fem::FunctionSpace& V0 = V.at(0).get();
  const fem::FunctionSpace& V1 = V.at(1).get();

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::runtime_error("Meshes are not the same.");
  const int tdim = mesh->topology().dim();

  assert(V0.element());
  assert(V1.element());
  if (V0.element()->hash() != V1.element()->hash())
    throw std::runtime_error("Function spaces must have the same element.");

  // Compute dof coordinates
  const xt::xtensor<double, 2> dof_coordinates
      = V1.tabulate_dof_coordinates(true);
  assert(dof_coordinates.shape(0) == 3);

  // Evaluate marker for each dof coordinate
  const xt::xtensor<bool, 1> marked_dofs = marker_fn(dof_coordinates);

  // Get dofmaps
  std::shared_ptr<const fem::DofMap> dofmap0 = V0.dofmap();
  assert(dofmap0);
  const int bs0 = dofmap0->bs();
  std::shared_ptr<const fem::DofMap> dofmap1 = V1.dofmap();
  assert(dofmap1);
  const int bs1 = dofmap1->bs();

  const int element_bs = dofmap0->element_dof_layout->block_size();
  assert(element_bs == dofmap1->element_dof_layout->block_size());

  // Iterate over cells
  const mesh::Topology& topology = mesh->topology();
  std::vector<std::array<std::int32_t, 2>> bc_dofs;
  for (int c = 0; c < topology.connectivity(tdim, 0)->num_nodes(); ++c)
  {
    // Get cell dofmaps
    auto cell_dofs0 = dofmap0->cell_dofs(c);
    auto cell_dofs1 = dofmap1->cell_dofs(c);

    // Loop over cell dofs and add to bc_dofs if marked.
    for (std::size_t i = 0; i < cell_dofs1.size(); ++i)
    {
      if (marked_dofs[cell_dofs1[i]])
      {
        // Unroll over blocks
        for (int k = 0; k < element_bs; ++k)
        {
          const int local_pos = element_bs * i + k;
          const std::div_t pos0 = std::div(local_pos, bs0);
          const std::div_t pos1 = std::div(local_pos, bs1);
          const std::int32_t dof_index0
              = bs0 * cell_dofs0[pos0.quot] + pos0.rem;
          const std::int32_t dof_index1
              = bs1 * cell_dofs1[pos1.quot] + pos1.rem;

          bc_dofs.push_back({dof_index0, dof_index1});
        }
      }
    }
  }

  // Remove duplicates
  std::sort(bc_dofs.begin(), bc_dofs.end());
  bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());

  // Copy to separate array
  std::array dofs = {std::vector<std::int32_t>(bc_dofs.size()),
                     std::vector<std::int32_t>(bc_dofs.size())};
  for (std::size_t i = 0; i < bc_dofs.size(); ++i)
  {
    dofs[0][i] = bc_dofs[i][0];
    dofs[1][i] = bc_dofs[i][1];
  }

  return dofs;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> fem::locate_dofs_geometrical(
    const fem::FunctionSpace& V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker_fn)
{
  // FIXME: Calling V.tabulate_dof_coordinates() is very expensive,
  // especially when we usually want the boundary dofs only. Add
  // interface that computes dofs coordinates only for specified cell.

  // Compute dof coordinates
  const xt::xtensor<double, 2> dof_coordinates
      = V.tabulate_dof_coordinates(true);
  assert(dof_coordinates.shape(0) == 3);

  // Compute marker for each dof coordinate
  const xt::xtensor<bool, 1> marked_dofs = marker_fn(dof_coordinates);

  std::vector<std::int32_t> dofs;
  dofs.reserve(std::count(marked_dofs.begin(), marked_dofs.end(), true));
  for (std::size_t i = 0; i < marked_dofs.size(); ++i)
  {
    if (marked_dofs[i])
      dofs.push_back(i);
  }

  return dofs;
}
//-----------------------------------------------------------------------------
