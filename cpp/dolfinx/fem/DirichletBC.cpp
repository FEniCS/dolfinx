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
#include <dolfinx/common/sort.h>
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

/// @brief Find the cell (local to process) and index of an entity
/// (local to cell) for a list of entities.
/// @param[in] mesh The mesh
/// @param[in] entities The list of entities
/// @param[in] dim The dimension of the entities
/// @returns A list of (cell_index, entity_index) pairs for each input
/// entity.
std::vector<std::pair<std::int32_t, int>>
find_local_entity_index(const mesh::Topology& topology,
                        std::span<const std::int32_t> entities, int dim)
{
  // Initialise entity-cell connectivity
  const int tdim = topology.dim();
  auto e_to_c = topology.connectivity(dim, tdim);
  if (!e_to_c)
  {
    throw std::runtime_error(
        "Entity-to-cell connectivity has not been computed. Missing dims "
        + std::to_string(dim) + "->" + std::to_string(tdim));
  }

  auto c_to_e = topology.connectivity(tdim, dim);
  if (!c_to_e)
  {
    throw std::runtime_error(
        "Cell-to-entity connectivity has not been computed. Missing dims "
        + std::to_string(tdim) + "->" + std::to_string(dim));
  }

  std::vector<std::pair<std::int32_t, int>> entity_indices;
  entity_indices.reserve(entities.size());
  for (std::int32_t e : entities)
  {
    // Get first attached cell
    assert(e_to_c->num_links(e) > 0);
    const int cell = e_to_c->links(e).front();

    // Get local index of facet with respect to the cell
    auto entities_d = c_to_e->links(cell);
    auto it = std::find(entities_d.begin(), entities_d.end(), e);
    assert(it != entities_d.end());
    std::size_t entity_local_index = std::distance(entities_d.begin(), it);
    entity_indices.emplace_back(cell, entity_local_index);
  }

  return entity_indices;
}
//-----------------------------------------------------------------------------

/// Find all DOFs on this process that have been detected on another
/// process
/// @param[in] comm A symmetric communicator
/// @param[in] map The index map with the dof layout
/// @param[in] bs_map The block size of the index map, i.e. the dof
/// array. It should be set to 1 if `dofs_local` contains block indices.
/// @param[in] dofs_local List of degrees of freedom on this rank
/// @returns Degrees of freedom found on the other ranks that exist on
/// this rank
std::vector<std::int32_t>
get_remote_dofs(MPI_Comm comm, const common::IndexMap& map, int bs_map,
                std::span<const std::int32_t> dofs_local)
{
  int num_neighbors(-1);
  {
    int outdegree(-2), weighted(-1);
    MPI_Dist_graph_neighbors_count(comm, &num_neighbors, &outdegree, &weighted);
    assert(num_neighbors == outdegree);
  }

  // Return early if there are no neighbors
  if (num_neighbors == 0)
    return {};

  // Figure out how many entries to receive from each neighbor
  const int num_dofs_block = dofs_local.size();
  std::vector<int> num_dofs_recv(num_neighbors);
  MPI_Request request;
  MPI_Ineighbor_allgather(&num_dofs_block, 1, MPI_INT, num_dofs_recv.data(), 1,
                          MPI_INT, comm, &request);

  std::vector<std::int64_t> dofs_global(dofs_local.size());
  if (bs_map == 1)
  {
    dofs_global.resize(dofs_local.size());
    map.local_to_global(dofs_local, dofs_global);
  }
  else
  {
    // Convert dofs indices to 'block' map indices
    std::vector<std::int32_t> dofs_local_m;
    dofs_local_m.reserve(dofs_local.size());
    std::transform(dofs_local.begin(), dofs_local.end(),
                   std::back_inserter(dofs_local_m),
                   [bs_map](auto dof) { return dof / bs_map; });

    // Compute global index of each block
    map.local_to_global(dofs_local_m, dofs_global);

    // Add offset
    std::transform(dofs_global.begin(), dofs_global.end(), dofs_local.begin(),
                   dofs_global.begin(),
                   [bs_map](auto global_block, auto local_dof)
                   { return bs_map * global_block + (local_dof % bs_map); });
  }

  MPI_Wait(&request, MPI_STATUS_IGNORE);

  // Compute displacements for data to receive. Last entry has total
  // number of received items.
  std::vector<int> disp(num_neighbors + 1, 0);
  std::partial_sum(num_dofs_recv.begin(), num_dofs_recv.end(),
                   std::next(disp.begin()));

  // NOTE: We could consider only dofs that we know are shared and use
  // MPI_Neighbor_alltoallv to send only to relevant processes.
  // Send/receive global index of dofs with bcs to all neighbors
  std::vector<std::int64_t> dofs_received(disp.back());
  MPI_Ineighbor_allgatherv(dofs_global.data(), dofs_global.size(), MPI_INT64_T,
                           dofs_received.data(), num_dofs_recv.data(),
                           disp.data(), MPI_INT64_T, comm, &request);

  // FIXME: check that dofs is sorted
  // Build vector of local dof indices that have been marked by another
  // process
  const std::array<std::int64_t, 2> range = map.local_range();
  std::span ghosts = map.ghosts();

  // Build map from ghost global index to local position
  // NOTE: Should we use map here or just one vector with ghosts and
  // std::distance?
  std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts;
  const std::int32_t local_size = range[1] - range[0];
  for (std::size_t i = 0; i < ghosts.size(); ++i)
    global_local_ghosts.emplace_back(ghosts[i], i + local_size);
  std::map<std::int64_t, std::int32_t> global_to_local(
      global_local_ghosts.begin(), global_local_ghosts.end());

  MPI_Wait(&request, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> dofs;
  for (auto dof_global : dofs_received)
  {
    // Insert owned dofs, else search in ghosts
    std::int64_t block = dof_global / bs_map;
    if (block >= range[0] and block < range[1])
      dofs.push_back(dof_global - bs_map * range[0]);
    else
    {
      if (auto it = global_to_local.find(block); it != global_to_local.end())
      {
        int offset = dof_global % bs_map;
        dofs.push_back(bs_map * it->second + offset);
      }
    }
  }

  return dofs;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::vector<std::int32_t> fem::locate_dofs_topological(
    const mesh::Topology& topology, const DofMap& dofmap, int dim,
    std::span<const std::int32_t> entities, bool remote)
{
  mesh::CellType cell_type = topology.cell_type();

  // Prepare an element - local dof layout for dofs on entities of the
  // entity_dim
  const int num_cell_entities = mesh::cell_num_entities(cell_type, dim);
  std::vector<std::vector<int>> entity_dofs;
  for (int i = 0; i < num_cell_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap.element_dof_layout().entity_closure_dofs(dim, i));
  }

  // Get cell index and local entity index
  std::vector<std::pair<std::int32_t, int>> entity_indices
      = find_local_entity_index(topology, entities, dim);

  std::vector<std::int32_t> dofs;
  dofs.reserve(entities.size()
               * dofmap.element_dof_layout().num_entity_closure_dofs(dim));

  // V is a sub space we need to take the block size of the dofmap and
  // the index map into account as they can differ
  const int bs = dofmap.bs();
  const int element_bs = dofmap.element_dof_layout().block_size();

  // Iterate over marked facets
  if (element_bs == bs)
  {
    // Work with blocks
    for (auto [cell, entity_local_index] : entity_indices)
    {
      // Get cell dofmap and loop over entity dofs
      auto cell_dofs = dofmap.cell_dofs(cell);
      for (int index : entity_dofs[entity_local_index])
        dofs.push_back(cell_dofs[index]);
    }
  }
  else if (bs == 1)
  {
    // Space is not blocked, unroll dofs
    for (auto [cell, entity_local_index] : entity_indices)
    {
      // Get cell dofmap and loop over facet dofs and 'unpack' blocked
      // dofs
      std::span<const std::int32_t> cell_dofs = dofmap.cell_dofs(cell);
      for (int index : entity_dofs[entity_local_index])
      {
        for (int k = 0; k < element_bs; ++k)
        {
          const std::div_t pos = std::div(element_bs * index + k, bs);
          dofs.push_back(bs * cell_dofs[pos.quot] + pos.rem);
        }
      }
    }
  }
  else
    throw std::runtime_error("Block size combination not supported");

  // TODO: is removing duplicates at this point worth the effort?
  // Remove duplicates
  std::ranges::sort(dofs);
  dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());

  if (remote)
  {
    // Get bc dof indices (local) in V spaces on this process that were
    // found by other processes, e.g. a vertex dof on this process that
    // has no connected facets on the boundary.
    auto map = dofmap.index_map;
    assert(map);

    // Create 'symmetric' neighbourhood communicator
    MPI_Comm comm;
    {
      std::span src = map->src();
      std::span dest = map->dest();
      std::vector<int> ranks;
      std::set_union(src.begin(), src.end(), dest.begin(), dest.end(),
                     std::back_inserter(ranks));
      ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
      MPI_Dist_graph_create_adjacent(
          map->comm(), ranks.size(), ranks.data(), MPI_UNWEIGHTED, ranks.size(),
          ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
    }

    std::vector<std::int32_t> dofs_remote;
    if (int map_bs = dofmap.index_map_bs(); map_bs == bs)
      dofs_remote = get_remote_dofs(comm, *map, 1, dofs);
    else
      dofs_remote = get_remote_dofs(comm, *map, map_bs, dofs);

    MPI_Comm_free(&comm);

    // Add received bc indices to dofs_local, sort, and remove
    // duplicates
    dofs.insert(dofs.end(), dofs_remote.begin(), dofs_remote.end());
    std::ranges::sort(dofs);
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
  }

  return dofs;
}
//-----------------------------------------------------------------------------
std::array<std::vector<std::int32_t>, 2> fem::locate_dofs_topological(
    const mesh::Topology& topology,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps, const int dim,
    std::span<const std::int32_t> entities, bool remote)
{
  // Get dofmaps
  const DofMap& dofmap0 = dofmaps.at(0).get();
  const DofMap& dofmap1 = dofmaps.at(1).get();

  // Check that dof layouts are the same
  assert(dofmap0.element_dof_layout() == dofmap1.element_dof_layout());

  mesh::CellType cell_type = topology.cell_type();

  // Build vector of local dofs for each cell entity
  const int num_cell_entities = mesh::cell_num_entities(cell_type, dim);
  std::vector<std::vector<int>> entity_dofs;
  for (int i = 0; i < num_cell_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap0.element_dof_layout().entity_closure_dofs(dim, i));
  }

  const std::array bs = {dofmap0.bs(), dofmap1.bs()};

  // Get cell index and local entity index
  std::vector<std::pair<std::int32_t, int>> entity_indices
      = find_local_entity_index(topology, entities, dim);

  // Iterate over marked facets
  const int element_bs = dofmap0.element_dof_layout().block_size();
  std::array<std::vector<std::int32_t>, 2> bc_dofs;
  bc_dofs[0].reserve(entities.size()
                     * dofmap0.element_dof_layout().num_entity_closure_dofs(dim)
                     * element_bs);
  bc_dofs[1].reserve(entities.size()
                     * dofmap0.element_dof_layout().num_entity_closure_dofs(dim)
                     * element_bs);
  for (auto [cell, entity_local_index] : entity_indices)
  {
    // Get cell dofmap
    std::span<const std::int32_t> cell_dofs0 = dofmap0.cell_dofs(cell);
    std::span<const std::int32_t> cell_dofs1 = dofmap1.cell_dofs(cell);
    assert(bs[0] * cell_dofs0.size() == bs[1] * cell_dofs1.size());

    // Loop over facet dofs and 'unpack' blocked dofs
    for (int index : entity_dofs[entity_local_index])
    {
      for (int k = 0; k < element_bs; ++k)
      {
        const int local_pos = element_bs * index + k;
        const std::div_t pos0 = std::div(local_pos, bs[0]);
        const std::div_t pos1 = std::div(local_pos, bs[1]);
        std::int32_t dof_index0 = bs[0] * cell_dofs0[pos0.quot] + pos0.rem;
        std::int32_t dof_index1 = bs[1] * cell_dofs1[pos1.quot] + pos1.rem;
        bc_dofs[0].push_back(dof_index0);
        bc_dofs[1].push_back(dof_index1);
      }
    }
  }

  // TODO: is removing duplicates at this point worth the effort?
  // Remove duplicates
  std::vector<std::int32_t> perm(bc_dofs[0].size());
  std::iota(perm.begin(), perm.end(), 0);
  dolfinx::argsort_radix<std::int32_t>(bc_dofs[0], perm);
  std::array<std::vector<std::int32_t>, 2> sorted_bc_dofs = bc_dofs;
  for (std::size_t b = 0; b < 2; ++b)
  {
    std::transform(perm.cbegin(), perm.cend(), sorted_bc_dofs[b].begin(),
                   [&bc_dofs = bc_dofs[b]](auto p) { return bc_dofs[p]; });
    sorted_bc_dofs[b].erase(
        std::unique(sorted_bc_dofs[b].begin(), sorted_bc_dofs[b].end()),
        sorted_bc_dofs[b].end());
  }

  if (!remote)
    return sorted_bc_dofs;
  else
  {
    // Get bc dof indices (local) for each of spaces on this process that
    // were found by other processes, e.g. a vertex dof on this process
    // that has no connected facets on the boundary.

    auto map0 = dofmap0.index_map;
    assert(map0);

    // Create 'symmetric' neighbourhood communicator
    MPI_Comm comm;
    {
      std::span src = map0->src();
      std::span dest = map0->dest();
      std::vector<int> ranks;
      std::set_union(src.begin(), src.end(), dest.begin(), dest.end(),
                     std::back_inserter(ranks));
      ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
      MPI_Dist_graph_create_adjacent(map0->comm(), ranks.size(), ranks.data(),
                                     MPI_UNWEIGHTED, ranks.size(), ranks.data(),
                                     MPI_UNWEIGHTED, MPI_INFO_NULL, false,
                                     &comm);
    }

    std::vector<std::int32_t> dofs_remote = get_remote_dofs(
        comm, *map0, dofmap0.index_map_bs(), sorted_bc_dofs[0]);

    // Add received bc indices to dofs_local
    sorted_bc_dofs[0].insert(sorted_bc_dofs[0].end(), dofs_remote.begin(),
                             dofs_remote.end());

    dofs_remote = get_remote_dofs(comm, *(dofmap1.index_map),
                                  dofmap1.index_map_bs(), sorted_bc_dofs[1]);
    sorted_bc_dofs[1].insert(sorted_bc_dofs[1].end(), dofs_remote.begin(),
                             dofs_remote.end());
    assert(sorted_bc_dofs[0].size() == sorted_bc_dofs[1].size());

    MPI_Comm_free(&comm);

    // Remove duplicates and sort
    perm.resize(sorted_bc_dofs[0].size());
    std::iota(perm.begin(), perm.end(), 0);
    dolfinx::argsort_radix<std::int32_t>(sorted_bc_dofs[0], perm);
    std::array<std::vector<std::int32_t>, 2> out_dofs = sorted_bc_dofs;
    for (std::size_t b = 0; b < 2; ++b)
    {
      std::transform(perm.cbegin(), perm.cend(), out_dofs[b].begin(),
                     [&sorted_dofs = sorted_bc_dofs[b]](auto p)
                     { return sorted_dofs[p]; });
      out_dofs[b].erase(std::unique(out_dofs[b].begin(), out_dofs[b].end()),
                        out_dofs[b].end());
    }

    assert(out_dofs[0].size() == out_dofs[1].size());
    return out_dofs;
  }
}
//-----------------------------------------------------------------------------
