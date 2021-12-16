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
/// Create a symmetric MPI neighbourhood communciator from an
/// input neighbourhood communicator
dolfinx::MPI::Comm create_symmetric_comm(MPI_Comm comm)
{
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
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &comm_sym);

  return dolfinx::MPI::Comm(comm_sym, false);
}

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
    const int cell = e_to_c->links(e)[0];

    // Get local index of facet with respect to the cell
    auto entities_d = c_to_e->links(cell);
    auto it = std::find(entities_d.begin(), entities_d.end(), e);
    assert(it != entities_d.end());
    const int entity_local_index = std::distance(entities_d.begin(), it);
    entity_indices.push_back({cell, entity_local_index});
  }
  return entity_indices;
};

//-----------------------------------------------------------------------------

/// Find all DOFs on this process that has been detected on another process
/// @param[in] comm A symmetric communicator based on the forward
/// neighborhood communicator in the IndexMap
/// @param[in] map The IndexMap with the dof layout
/// @param[in] bs The block size of the dof array
/// @param[in] dofs_local List of degrees of freedom local to process
/// (unrolled). It might contain indices not found on other processes
/// @returns List of degrees of freedom that was found on the other processes
/// that are in the local range (including ghosts)
std::vector<std::int32_t>
get_remote_dofs(MPI_Comm comm, const common::IndexMap& map, int bs_map, int bs,
                const xtl::span<const std::int32_t>& dofs_local)
{
  int num_neighbors(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &num_neighbors, &outdegree, &weighted);
  assert(num_neighbors == outdegree);

  // Return early if there are no neighbors
  if (num_neighbors == 0)
    return {};

  // Figure out how many entries to receive from each neighbor
  assert(dofs_local.size() % bs == 0);
  const int num_dofs_block = dofs_local.size() / bs;
  std::vector<int> num_dofs_recv(num_neighbors);
  MPI_Request request;
  MPI_Ineighbor_allgather(&num_dofs_block, 1, MPI_INT, num_dofs_recv.data(), 1,
                          MPI_INT, comm, &request);

  // Map local dof block indices to global dof block indices
  std::vector<std::int64_t> dofs_global(num_dofs_block);
  if (bs_map == 1 and bs == 1)
    map.local_to_global(dofs_local, dofs_global);
  else
  {
    // Convert dofs indices to 'block' indices
    std::vector<std::int32_t> dofs_local_block;
    dofs_local_block.reserve(num_dofs_block);
    for (std::size_t i = 0; i < dofs_local.size(); i += bs)
      dofs_local_block.push_back((dofs_local[i] / bs));

    // Compute global index of each block
    map.local_to_global(dofs_local_block, dofs_global);
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

  // Build map from ghost to local position
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
  for (auto dof_global_block : dofs_received)
  {
    for (int k = 0; k < bs; ++k)
    {
      // Insert owned dofs, else search in ghosts
      if (dof_global_block >= bs_map * range[0]
          and dof_global_block < bs_map * range[1])
        dofs.push_back(bs * dof_global_block + k - bs_map * range[0]);
      else
      {
        if (auto it = global_to_local.find(dof_global_block);
            it != global_to_local.end())
        {
          dofs.push_back(bs * it->second + k);
        }
      }
    }
  }

  return dofs;
}
//-----------------------------------------------------------------------------
} // namespace

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
  if (V0.element()->hash() != V1.element()->hash())
    throw std::runtime_error("Function spaces must have the same element.");

  // Get dofmaps
  std::shared_ptr<const DofMap> dofmap0 = V0.dofmap();
  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
  assert(dofmap0);
  assert(dofmap1);

  // Check that dof layouts are the same
  assert(dofmap0->element_dof_layout);
  assert(dofmap1->element_dof_layout);
  assert(*dofmap0->element_dof_layout.get()
         == *dofmap1->element_dof_layout.get());

  // Build vector of local dofs for each cell entity
  const int num_cell_entities
      = mesh::cell_num_entities(mesh->topology().cell_type(), dim);
  std::vector<std::vector<int>> entity_dofs;
  entity_dofs.reserve(num_cell_entities);
  for (int i = 0; i < num_cell_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap0->element_dof_layout->entity_closure_dofs(dim, i));
  }

  const int bs0 = dofmap0->bs();
  const int bs1 = dofmap1->bs();

  // Get cell index and local entity index
  std::vector<std::pair<std::int32_t, int>> entity_indices
      = find_local_entity_index(mesh, entities, dim);

  // Iterate over marked facets
  const int num_entity_dofs
      = dofmap0->element_dof_layout->num_entity_closure_dofs(dim);
  const int element_bs = dofmap0->element_dof_layout->block_size();

  std::array<std::vector<std::int32_t>, 2> bc_dofs;
  bc_dofs[0].reserve(entities.size() * num_entity_dofs * element_bs);
  bc_dofs[1].reserve(entities.size() * num_entity_dofs * element_bs);
  for (auto [cell, entity_local_index] : entity_indices)
  {
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
    for (std::size_t i = 0; i < bc_dofs[1].size(); ++i)
      sorted_bc_dofs[b][i] = bc_dofs[b][perm[i]];
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
    dolfinx::MPI::Comm comm = create_symmetric_comm(
        V0.dofmap()->index_map->comm(common::IndexMap::Direction::forward));

    if (V0.dofmap()->index_map_bs() < bs0)
    {
      throw std::runtime_error(
          "Different IndexMap/dofmap block sizes is not supported.");
    }
    std::vector<std::int32_t> dofs_remote
        = get_remote_dofs(comm.comm(), *V0.dofmap()->index_map, bs0,
                          V0.dofmap()->index_map_bs(), sorted_bc_dofs[0]);

    // Add received bc indices to dofs_local
    sorted_bc_dofs[0].insert(sorted_bc_dofs[0].end(), dofs_remote.begin(),
                             dofs_remote.end());

    if (V1.dofmap()->index_map_bs() < bs1)
    {
      throw std::runtime_error(
          "Different IndexMap/dofmap block sizes is not supported.");
    }
    dofs_remote
        = get_remote_dofs(comm.comm(), *V1.dofmap()->index_map,
                          V1.dofmap()->index_map_bs(), bs1, sorted_bc_dofs[1]);
    sorted_bc_dofs[1].insert(sorted_bc_dofs[1].end(), dofs_remote.begin(),
                             dofs_remote.end());
    assert(sorted_bc_dofs[0].size() == sorted_bc_dofs[1].size());

    // Remove duplicates and sort
    perm.resize(sorted_bc_dofs[0].size());
    std::iota(perm.begin(), perm.end(), 0);
    dolfinx::argsort_radix<std::int32_t>(sorted_bc_dofs[0], perm);

    std::array<std::vector<std::int32_t>, 2> out_dofs = sorted_bc_dofs;
    for (std::size_t b = 0; b < 2; ++b)
    {
      for (std::size_t i = 0; i < sorted_bc_dofs[1].size(); ++i)
        out_dofs[b][i] = sorted_bc_dofs[b][perm[i]];
      out_dofs[b].erase(std::unique(out_dofs[b].begin(), out_dofs[b].end()),
                        out_dofs[b].end());
    }
    assert(out_dofs[0].size() == out_dofs[1].size());
    return out_dofs;
  }
}
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
  entity_dofs.reserve(num_cell_entities);
  for (int i = 0; i < num_cell_entities; ++i)
  {
    entity_dofs.push_back(
        dofmap->element_dof_layout->entity_closure_dofs(dim, i));
  }

  const int num_entity_closure_dofs
      = dofmap->element_dof_layout->num_entity_closure_dofs(dim);
  std::vector<std::int32_t> dofs;
  dofs.reserve(entities.size() * num_entity_closure_dofs);

  // Get cell index and local entity index
  std::vector<std::pair<std::int32_t, int>> entity_indices
      = find_local_entity_index(mesh, entities, dim);

  // If space is not a sub space we not not need to take the block size
  // into account
  if (V.component().empty())
  {
    for (auto [cell, entity_local_index] : entity_indices)
    {
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
      auto map = V.dofmap()->index_map;
      dolfinx::MPI::Comm comm = create_symmetric_comm(
          map->comm(common::IndexMap::Direction::forward));
      const std::vector<std::int32_t> dofs_remote
          = get_remote_dofs(comm.comm(), *map, 1, 1, dofs);

      // Add received bc indices to dofs_local
      dofs.insert(dofs.end(), dofs_remote.begin(), dofs_remote.end());

      // Remove duplicates
      std::sort(dofs.begin(), dofs.end());
      dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    }
  }
  else
  {
    // V is a sub space we need to take the block size of the dofmap and
    // index map into account, as they differ
    const int bs = dofmap->bs();
    const int element_bs = dofmap->element_dof_layout->block_size();

    // Iterate over marked facets
    for (auto [cell, entity_local_index] : entity_indices)
    {

      // Get cell dofmap
      xtl::span<const std::int32_t> cell_dofs = dofmap->cell_dofs(cell);

      // Loop over facet dofs and 'unpack' blocked dofs
      for (int i = 0; i < num_entity_closure_dofs; ++i)
      {
        const int index = entity_dofs[entity_local_index][i];
        for (int k = 0; k < element_bs; ++k)
        {
          const std::div_t pos = std::div(element_bs * index + k, bs);
          dofs.push_back(bs * cell_dofs[pos.quot] + pos.rem);
        }
      }
    }

    // TODO: is removing duplicates at this point worth the effort?
    // Remove duplicates
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());

    if (remote)
    {
      // Get bc dof indices (local) in (V, Vg) spaces on this process that
      // were found by other processes, e.g. a vertex dof on this process
      // that has no connected facets on the boundary.
      dolfinx::MPI::Comm comm = create_symmetric_comm(
          V.dofmap()->index_map->comm(common::IndexMap::Direction::forward));

      if (V.dofmap()->index_map_bs() < V.dofmap()->bs())
      {
        throw std::runtime_error(
            "Different IndexMap/dofmap block sizes is not supported.");
      }
      const std::vector<std::int32_t> dofs_remote
          = get_remote_dofs(comm.comm(), *V.dofmap()->index_map,
                            V.dofmap()->index_map_bs(), bs, dofs);

      // Add received bc indices to dofs_local
      dofs.insert(dofs.end(), dofs_remote.begin(), dofs_remote.end());

      // Remove duplicates and sort
      std::sort(dofs.begin(), dofs.end());
      dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    }
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
  if (V0.element()->hash() != V1.element()->hash())
    throw std::runtime_error("Function spaces must have the same element.");

  // Compute dof coordinates
  const xt::xtensor<double, 2> dof_coordinates
      = V1.tabulate_dof_coordinates(true);
  assert(dof_coordinates.shape(0) == 3);

  // Evaluate marker for each dof coordinate
  const xt::xtensor<bool, 1> marked_dofs = marker_fn(dof_coordinates);

  // Get dofmaps
  std::shared_ptr<const DofMap> dofmap0 = V0.dofmap();
  assert(dofmap0);
  const int bs0 = dofmap0->bs();
  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
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
