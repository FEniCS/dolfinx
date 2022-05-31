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
//-----------------------------------------------------------------------------
/// Find the cell (local to process) and index of an entity (local to cell) for
/// a list of entities
/// @param[in] mesh The mesh
/// @param[in] entities The list of entities
/// @param[in] dim The dimension of the entities
/// @returns A list of (cell_index, entity_index) pairs for each input entity
std::vector<std::pair<std::int32_t, int>>
find_local_entity_index(std::shared_ptr<const mesh::Mesh> mesh,
                        const xtl::span<const std::int32_t>& entities,
                        const int dim)
{
  // Initialise entity-cell connectivity
  const int tdim = mesh->topology().dim();
  mesh->topology_mutable().create_entities(tdim);
  mesh->topology_mutable().create_connectivity(dim, tdim);
  auto e_to_c = mesh->topology().connectivity(dim, tdim);
  assert(e_to_c);
  auto c_to_e = mesh->topology().connectivity(tdim, dim);
  assert(c_to_e);

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
                const xtl::span<const std::int32_t>& dofs_local)
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
    std::transform(dofs_local.cbegin(), dofs_local.cend(),
                   std::back_inserter(dofs_local_m),
                   [bs_map](auto dof) { return dof / bs_map; });

    // Compute global index of each block
    map.local_to_global(dofs_local_m, dofs_global);

    // Add offset
    std::transform(dofs_global.cbegin(), dofs_global.cend(),
                   dofs_local.cbegin(), dofs_global.begin(),
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
  // Build vector of local dof indicies that have been marked by another
  // process
  const std::array<std::int64_t, 2> range = map.local_range();
  const std::vector<std::int64_t>& ghosts = map.ghosts();

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
std::vector<std::int32_t>
fem::locate_dofs_topological(const FunctionSpace& V, int dim,
                             const xtl::span<const std::int32_t>& entities,
                             bool remote)
{
  assert(V.dofmap());
  std::shared_ptr<const DofMap> dofmap = V.dofmap();
  assert(V.mesh());
  std::shared_ptr<const mesh::Mesh> mesh = V.mesh();

  // Prepare an element - local dof layout for dofs on entities of the
  // entity_dim
  const int num_cell_entities
      = mesh::cell_num_entities(mesh->topology().cell_type(), dim);
  std::vector<std::vector<int>> entity_dofs;
  for (int i = 0; i < num_cell_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap->element_dof_layout().entity_closure_dofs(dim, i));
  }

  // Get cell index and local entity index
  std::vector<std::pair<std::int32_t, int>> entity_indices
      = find_local_entity_index(mesh, entities, dim);

  std::vector<std::int32_t> dofs;
  dofs.reserve(entities.size()
               * dofmap->element_dof_layout().num_entity_closure_dofs(dim));

  // V is a sub space we need to take the block size of the dofmap and
  // the index map into account as they can differ
  const int bs = dofmap->bs();
  const int element_bs = dofmap->element_dof_layout().block_size();

  // Iterate over marked facets
  if (element_bs == bs)
  {
    // Work with blocks
    for (auto [cell, entity_local_index] : entity_indices)
    {
      // Get cell dofmap and loop over entity dofs
      auto cell_dofs = dofmap->cell_dofs(cell);
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
      xtl::span<const std::int32_t> cell_dofs = dofmap->cell_dofs(cell);
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
  std::sort(dofs.begin(), dofs.end());
  dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());

  if (remote)
  {
    // Get bc dof indices (local) in V spaces on this process that were
    // found by other processes, e.g. a vertex dof on this process that
    // has no connected facets on the boundary.
    auto map = dofmap->index_map;
    assert(map);

    // Create 'symmetric' neighbourhood communicator
    MPI_Comm comm;
    {
      const std::vector<int>& src = map->src();
      const std::vector<int>& dest = map->dest();

      std::vector<int> ranks;
      std::set_union(src.begin(), src.end(), dest.begin(), dest.end(),
                     std::back_inserter(ranks));
      ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

      MPI_Dist_graph_create_adjacent(
          map->comm(), ranks.size(), ranks.data(), MPI_UNWEIGHTED, ranks.size(),
          ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
    }

    std::vector<std::int32_t> dofs_remote;
    if (int map_bs = dofmap->index_map_bs(); map_bs == bs)
      dofs_remote = get_remote_dofs(comm, *map, 1, dofs);
    else
      dofs_remote = get_remote_dofs(comm, *map, map_bs, dofs);

    MPI_Comm_free(&comm);

    // Add received bc indices to dofs_local, sort, and remove
    // duplicates
    dofs.insert(dofs.end(), dofs_remote.begin(), dofs_remote.end());
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
  }

  return dofs;
}
//-----------------------------------------------------------------------------
std::array<std::vector<std::int32_t>, 2> fem::locate_dofs_topological(
    const std::array<std::reference_wrapper<const FunctionSpace>, 2>& V,
    const int dim, const xtl::span<const std::int32_t>& entities, bool remote)
{
  const FunctionSpace& V0 = V.at(0).get();
  const FunctionSpace& V1 = V.at(1).get();

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::runtime_error("Meshes are not the same.");

  // FIXME: Elements must be the same?
  assert(V0.element());
  assert(V1.element());
  if (*V0.element() != *V1.element())
    throw std::runtime_error("Function spaces must have the same element.");

  // Get dofmaps
  std::shared_ptr<const DofMap> dofmap0 = V0.dofmap();
  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
  assert(dofmap0);
  assert(dofmap1);

  // Check that dof layouts are the same
  assert(dofmap0->element_dof_layout() == dofmap1->element_dof_layout());

  // Build vector of local dofs for each cell entity
  const int num_cell_entities
      = mesh::cell_num_entities(mesh->topology().cell_type(), dim);
  std::vector<std::vector<int>> entity_dofs;
  for (int i = 0; i < num_cell_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap0->element_dof_layout().entity_closure_dofs(dim, i));
  }

  const std::array bs = {dofmap0->bs(), dofmap1->bs()};

  // Get cell index and local entity index
  std::vector<std::pair<std::int32_t, int>> entity_indices
      = find_local_entity_index(mesh, entities, dim);

  // Iterate over marked facets
  const int element_bs = dofmap0->element_dof_layout().block_size();
  std::array<std::vector<std::int32_t>, 2> bc_dofs;
  bc_dofs[0].reserve(
      entities.size()
      * dofmap0->element_dof_layout().num_entity_closure_dofs(dim)
      * element_bs);
  bc_dofs[1].reserve(
      entities.size()
      * dofmap0->element_dof_layout().num_entity_closure_dofs(dim)
      * element_bs);
  for (auto [cell, entity_local_index] : entity_indices)
  {
    // Get cell dofmap
    xtl::span<const std::int32_t> cell_dofs0 = dofmap0->cell_dofs(cell);
    xtl::span<const std::int32_t> cell_dofs1 = dofmap1->cell_dofs(cell);
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

    auto map0 = V0.dofmap()->index_map;
    assert(map0);

    // Create 'symmetric' neighbourhood communicator
    MPI_Comm comm;
    {
      const std::vector<int>& src = map0->src();
      const std::vector<int>& dest = map0->dest();

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
        comm, *map0, V0.dofmap()->index_map_bs(), sorted_bc_dofs[0]);

    // Add received bc indices to dofs_local
    sorted_bc_dofs[0].insert(sorted_bc_dofs[0].end(), dofs_remote.begin(),
                             dofs_remote.end());

    dofs_remote
        = get_remote_dofs(comm, *(V1.dofmap()->index_map),
                          V1.dofmap()->index_map_bs(), sorted_bc_dofs[1]);
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
std::vector<std::int32_t> fem::locate_dofs_geometrical(
    const FunctionSpace& V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker_fn)
{
  // FIXME: Calling V.tabulate_dof_coordinates() is very expensive,
  // especially when we usually want the boundary dofs only. Add
  // interface that computes dofs coordinates only for specified cell.

  assert(V.element());
  if (V.element()->is_mixed())
  {
    throw std::runtime_error(
        "Cannot locate dofs geometrically for mixed space. Use subspaces.");
  }

  // Compute dof coordinates
  const std::vector<double> dof_coordinates = V.tabulate_dof_coordinates(true);

  // Compute marker for each dof coordinate
  auto x = xt::adapt(dof_coordinates,
                     std::vector<std::size_t>{3, dof_coordinates.size() / 3});
  const xt::xtensor<bool, 1> marked_dofs = marker_fn(x);

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
std::array<std::vector<std::int32_t>, 2> fem::locate_dofs_geometrical(
    const std::array<std::reference_wrapper<const FunctionSpace>, 2>& V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker_fn)
{
  // FIXME: Calling V.tabulate_dof_coordinates() is very expensive,
  // especially when we usually want the boundary dofs only. Add
  // interface that computes dofs coordinates only for specified cell.

  // Get function spaces
  const FunctionSpace& V0 = V.at(0).get();
  const FunctionSpace& V1 = V.at(1).get();

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::runtime_error("Meshes are not the same.");
  const int tdim = mesh->topology().dim();

  assert(V0.element());
  assert(V1.element());
  if (*V0.element() != *V1.element())
    throw std::runtime_error("Function spaces must have the same element.");

  // Compute dof coordinates
  const std::vector<double> dof_coordinates = V1.tabulate_dof_coordinates(true);

  // Evaluate marker for each dof coordinate
  auto x = xt::adapt(dof_coordinates,
                     std::vector<std::size_t>{3, dof_coordinates.size() / 3});
  const xt::xtensor<bool, 1> marked_dofs = marker_fn(x);

  // Get dofmaps
  std::shared_ptr<const DofMap> dofmap0 = V0.dofmap();
  assert(dofmap0);
  const int bs0 = dofmap0->bs();
  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
  assert(dofmap1);
  const int bs1 = dofmap1->bs();

  const int element_bs = dofmap0->element_dof_layout().block_size();
  assert(element_bs == dofmap1->element_dof_layout().block_size());

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
  std::transform(bc_dofs.cbegin(), bc_dofs.cend(), dofs[0].begin(),
                 [](auto dof) { return dof[0]; });
  std::transform(bc_dofs.cbegin(), bc_dofs.cend(), dofs[1].begin(),
                 [](auto dof) { return dof[1]; });

  return dofs;
}
//-----------------------------------------------------------------------------
