// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dofmapbuilder.h"
#include "ElementDofLayout.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <cstdint>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{

namespace stdex = std::experimental;
template <typename T>
using mdspan2_t = stdex::mdspan<T, stdex::dextents<std::size_t, 2>>;

//-----------------------------------------------------------------------------

/// Build a graph for owned dofs and apply graph reordering function
/// @param[in] dofmap The local dofmap (cell -> dofs)
/// @param[in] owned_size Number of dofs owned by this process
/// @param[in] original_to_contiguous Map from dof indices in @p dofmap
/// to new indices that are ordered such that owned indices are [0,
/// owned_size)
/// @param[in] reorder_fn The graph reordering function to apply
/// @return Map from original_to_contiguous[i] to new index after
/// reordering
std::vector<int>
reorder_owned(mdspan2_t<const std::int32_t> dofmap, std::int32_t owned_size,
              const std::vector<int>& original_to_contiguous,
              const std::function<std::vector<int>(
                  const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  std::vector<std::int32_t> graph_data, graph_offsets;

  // Compute maximum number of graph out edges edges per dof
  std::vector<int> num_edges(owned_size);
  for (std::size_t cell = 0; cell < dofmap.extent(0); ++cell)
  {
    std::span<const std::int32_t> nodes(
        dofmap.data_handle() + cell * dofmap.extent(1), dofmap.extent(1));
    for (auto n0 : nodes)
    {
      const std::int32_t node_0 = original_to_contiguous[n0];

      // Skip unowned node
      if (node_0 >= owned_size)
        continue;
      for (auto n1 : nodes)
      {
        if (n0 != n1 and original_to_contiguous[n1] < owned_size)
          ++num_edges[node_0];
      }
    }
  }

  // Compute adjacency list with duplicate edges
  std::vector<std::int32_t> offsets(num_edges.size() + 1, 0);
  std::partial_sum(num_edges.begin(), num_edges.end(),
                   std::next(offsets.begin(), 1));
  std::vector<std::int32_t> edges(offsets.back());
  for (std::size_t cell = 0; cell < dofmap.extent(0); ++cell)
  {
    std::span<const std::int32_t> nodes(
        dofmap.data_handle() + cell * dofmap.extent(1), dofmap.extent(1));
    for (auto n0 : nodes)
    {
      const std::int32_t node_0 = original_to_contiguous[n0];
      if (node_0 >= owned_size)
        continue;
      for (auto n1 : nodes)
      {
        if (const std::int32_t node_1 = original_to_contiguous[n1];
            n0 != n1 and node_1 < owned_size)
        {
          edges[offsets[node_0]++] = node_1;
        }
      }
    }
  }

  // Eliminate duplicate edges and create AdjacencyList
  graph_offsets.resize(num_edges.size() + 1, 0);
  std::int32_t current_offset = 0;
  for (std::size_t i = 0; i < num_edges.size(); ++i)
  {
    std::sort(std::next(edges.begin(), current_offset),
              std::next(edges.begin(), current_offset + num_edges[i]));
    const auto it
        = std::unique(std::next(edges.begin(), current_offset),
                      std::next(edges.begin(), current_offset + num_edges[i]));
    graph_data.insert(graph_data.end(),
                      std::next(edges.begin(), current_offset), it);
    graph_offsets[i + 1]
        = graph_offsets[i]
          + std::distance(std::next(edges.begin(), current_offset), it);
    current_offset += num_edges[i];
  }

  // Re-order graph and return re-odering
  assert(reorder_fn);
  return reorder_fn(
      graph::AdjacencyList(std::move(graph_data), std::move(graph_offsets)));
}

//-----------------------------------------------------------------------------

/// Build a simple dofmap from ElementDofmap based on mesh entity
/// indices (local and global)
///
/// @param [in] mesh The mesh to build the dofmap on
/// @param [in] topology The mesh topology
/// @param [in] element_dof_layout The layout of dofs on a cell
/// @return Returns {dofmap (local to the process), local-to-global map
/// to get the global index of local dof i, dof indices, vector of
/// {dimension, mesh entity index} for each local dof i}
std::tuple<std::vector<std::int32_t>, std::vector<std::int64_t>,
           std::vector<std::pair<std::int8_t, std::int32_t>>>
build_basic_dofmap(
    const mesh::Topology& topology,
    const std::vector<fem::ElementDofLayout>& element_dof_layouts)
{
  // Start timer for dofmap initialization
  common::Timer t0("Init dofmap from element dofmap");

  // Topological dimension
  const int D = topology.dim();

  // Checks for mixed topology
  if (element_dof_layouts.size() != topology.cell_types().size())
    throw std::runtime_error("Mixed topology mismatch: cell types");

  // Should be 2*n+1 cell group offsets in Topology for n elements
  if (element_dof_layouts.size() * 2 + 1
      != topology.entity_group_offsets(D).size())
  {
    throw std::runtime_error("Mixed topology mismatch: groups");
  }

  // Mixed topology can only manage one dof (e.g. on vertex, or on edge,
  // i.e. P1 or P2) for now.
  if (element_dof_layouts.size() > 1)
  {
    for (auto e : element_dof_layouts)
    {
      for (int d = 0; d <= D; ++d)
      {
        if (e.num_entity_dofs(d) > 1)
          throw std::runtime_error("Mixed topology: dofmapbuilder does not yet "
                                   "support elements with more than "
                                   "one dof per entity dimension (P1/P2)");
      }
    }

    std::array<int, 3> nd;
    for (int d = 0; d < D; ++d)
      nd[d] = element_dof_layouts[0].num_entity_dofs(d);
    for (int d = 0; d < D; ++d)
    {
      for (std::size_t i = 1; i < element_dof_layouts.size(); ++i)
      {
        if (element_dof_layouts[i].num_entity_dofs(d) != nd[d])
        {
          throw std::runtime_error(
              "Mixed topology: dofmapbuilder - incompatible elements "
              "(different number of dofs on shared entity)");
        }
      }
    }
  }

  // Generate and number required mesh entities
  std::vector<std::int8_t> needs_entities(D + 1, false);
  std::vector<std::int32_t> num_mesh_entities_local(D + 1, 0),
      num_mesh_entities_global(D + 1, 0);
  for (int d = 0; d <= D; ++d)
  {
    // FIXME: Mixed-topology - P2/Q2 - Q has dof on interior, P does not - need
    // to check all elements here.
    if (element_dof_layouts[0].num_entity_dofs(d) > 0)
    {
      if (!topology.connectivity(d, 0))
      {
        throw std::runtime_error(
            "Cannot create basic dofmap. Missing entities of dimension "
            + std::to_string(d) + " .");
      }
      needs_entities[d] = true;
      num_mesh_entities_local[d] = topology.connectivity(d, 0)->num_nodes();
      assert(topology.index_map(d));
      num_mesh_entities_global[d] = topology.index_map(d)->size_global();
    }
  }

  // Collect cell -> entity connectivities
  std::vector<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>>
      connectivity;
  for (int d = 0; d <= D; ++d)
    connectivity.push_back(topology.connectivity(D, d));

  // Build global dof arrays
  std::vector<std::vector<std::int64_t>> global_indices(D + 1);
  for (int d = 0; d <= D; ++d)
  {
    if (needs_entities[d])
    {
      auto map = topology.index_map(d);
      assert(map);
      global_indices[d] = map->global_indices();
    }
  }

  // Number of dofs on this process
  // FIXME: This assumes that the number of dofs on the entities of each element
  // must be the same. This must be true for entities which are shared
  // (vertices, edges, facets) but may differ for cell.
  std::int32_t local_size(0), d(0);
  for (std::int32_t n : num_mesh_entities_local)
    local_size += n * element_dof_layouts[0].num_entity_dofs(d++);

  // Allocate dofmap memory
  const int num_cells = topology.connectivity(D, 0)->num_nodes();
  const std::vector<int>& group_offsets = topology.entity_group_offsets(D);

  const std::size_t nelem = element_dof_layouts.size();
  std::vector<std::int32_t> dofs(num_cells * element_dof_layouts[0].num_dofs());

  // Allocate entity indices array
  std::vector<std::vector<int32_t>> entity_indices_local(D + 1);
  std::vector<std::vector<int64_t>> entity_indices_global(D + 1);
  for (int d = 0; d <= D; ++d)
  {
    for (auto cell_type : topology.cell_types())
    {
      const std::size_t num_entities = mesh::cell_num_entities(cell_type, d);
      entity_indices_local[d].resize(
          std::max(num_entities, entity_indices_local[d].size()));
      entity_indices_global[d].resize(entity_indices_local[d].size());
    }
  }

  // Entity dofs on cell (dof = entity_dofs[element][dim][entity][index])
  std::vector<std::vector<std::vector<std::vector<int>>>> entity_dofs(nelem);
  for (std::size_t i = 0; i < element_dof_layouts.size(); ++i)
    entity_dofs[i] = element_dof_layouts[i].entity_dofs_all();

  // Storage for local-to-global map
  std::vector<std::int64_t> local_to_global(local_size);

  // Dof -> (dim, entity index) marker
  std::vector<std::pair<std::int8_t, std::int32_t>> dof_entity(local_size);

  assert(group_offsets.back() == connectivity[0]->num_nodes());

  // Loop over cells, group by group, and build dofmaps from respective
  // ElementDofmap
  const int num_dofs_cell = element_dof_layouts[0].num_dofs();
  for (std::size_t i = 0; i < 2 * nelem; ++i)
  {
    for (int c = group_offsets[i]; c < group_offsets[i + 1]; ++c)
    {
      // Get local (process) and global indices for each cell entity
      for (int d = 0; d < D; ++d)
      {
        if (needs_entities[d])
        {
          auto entities = connectivity[d]->links(c);
          for (std::size_t i = 0; i < entities.size(); ++i)
          {
            entity_indices_local[d][i] = entities[i];
            entity_indices_global[d][i] = global_indices[d][entities[i]];
          }
        }
      }

      // Handle cell index separately because cell.entities(D) doesn't work.
      if (needs_entities[D])
      {
        entity_indices_global[D][0] = global_indices[D][c];
        entity_indices_local[D][0] = c;
      }

      std::span<std::int32_t> dofs_c(dofs.data() + c * num_dofs_cell,
                                     num_dofs_cell);

      // Iterate over each topological dimension for this element (twice, once
      // for regular, and later for ghosts).
      const int elem = i % nelem;
      std::int32_t offset_local = 0;
      for (auto e_dofs_d = entity_dofs[elem].begin();
           e_dofs_d != entity_dofs[elem].end(); ++e_dofs_d)
      {
        const std::size_t d
            = std::distance(entity_dofs[elem].begin(), e_dofs_d);

        // Iterate over each entity of current dimension d
        for (auto e_dofs = e_dofs_d->begin(); e_dofs != e_dofs_d->end();
             ++e_dofs)
        {
          // Get entity indices (local to cell, local to process, and
          // global)
          const std::size_t e = std::distance(e_dofs_d->begin(), e_dofs);
          const std::int32_t e_index_local = entity_indices_local[d][e];

          // Loop over dofs belong to entity e of dimension d (d, e)
          // d: topological dimension
          // e: local entity index
          // dof_local: local index of dof at (d, e)
          const std::size_t num_entity_dofs = e_dofs->size();
          for (auto dof_local = e_dofs->begin(); dof_local != e_dofs->end();
               ++dof_local)
          {
            std::size_t count = std::distance(e_dofs->begin(), dof_local);
            std::int32_t dof
                = offset_local + num_entity_dofs * e_index_local + count;
            dofs_c[*dof_local] = dof;
          }
        }
        offset_local
            += entity_dofs[elem][d][0].size() * num_mesh_entities_local[d];
      }
    }
  }

  // Create local to global map and dof entity map. NOTE this must be done
  // outside of the above loop as some processes may have vertices that don't
  // belong to a cell on that process.
  std::int32_t offset_local = 0;
  std::int64_t offset_global = 0;
  for (int d = 0; d <= D; ++d)
  {
    if (needs_entities[d])
    {
      // NOTE This assumes all entities have the same number of dofs
      // Mixed topology: probably OK for P1/Q1 but not higher...
      auto num_entity_dofs = entity_dofs[0][d][0].size();
      for (std::int32_t e_index_local = 0;
           e_index_local < num_mesh_entities_local[d]; ++e_index_local)
      {
        auto e_index_global = global_indices[d][e_index_local];

        for (std::size_t count = 0; count < num_entity_dofs; ++count)
        {
          const std::int32_t dof
              = offset_local + num_entity_dofs * e_index_local + count;
          local_to_global[dof]
              = offset_global + num_entity_dofs * e_index_global + count;
          dof_entity[dof] = {d, e_index_local};
        }
      }
      offset_local += num_entity_dofs * num_mesh_entities_local[d];
      offset_global += num_entity_dofs * num_mesh_entities_global[d];
    }
  }

  return {std::move(dofs), std::move(local_to_global), std::move(dof_entity)};
}
//-----------------------------------------------------------------------------

/// Compute re-ordering map from old local index to new local index. The
/// M dofs owned by this process are reordered for locality and fill the
/// positions [0, ..., M). Dof owned by another process are placed at
/// the end, i.e. in the positions [M, ..., N), where N is the total
/// number of dofs on this process.
///
/// @param [in] dofmap The basic dofmap data
/// @param [in] dof_entity Map from dof index to (dim, entity_index),
/// where entity_index is the process-wise mesh entity index
/// @param [in] topology The mesh topology
/// @param [in] reorder_fn Graph reordering function that is applied for
/// dof re-ordering
/// @return The pair (old-to-new local index map, M), where M is the
/// number of dofs owned by this process
std::pair<std::vector<std::int32_t>, std::int32_t> compute_reordering_map(
    mdspan2_t<const std::int32_t> dofmap,
    const std::vector<std::pair<std::int8_t, std::int32_t>>& dof_entity,
    const mesh::Topology& topology,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  common::Timer t0("Compute dof reordering map");

  // Get mesh entity ownership offset for each topological dimension
  const int D = topology.dim();
  std::vector<std::int32_t> offset(D + 1, -1);
  for (std::size_t d = 0; d < offset.size(); ++d)
  {
    auto map = topology.index_map(d);
    if (map)
      offset[d] = map->size_local();
  }

  // Compute the number of dofs 'owned' by this process
  const std::int32_t owned_size = std::accumulate(
      dof_entity.begin(), dof_entity.end(), std::int32_t(0),
      [&offset = std::as_const(offset)](std::int32_t a, auto b)
      { return b.second < offset[b.first] ? a + 1 : a; });

  // Re-order dofs, increasing local dof index by iterating over cells

  // Create map from old index to new contiguous numbering for locally
  // owned dofs. Set to -1 for unowned dofs.
  std::vector<int> original_to_contiguous(dof_entity.size(), -1);
  std::int32_t counter_owned(0), counter_unowned(owned_size);
  for (std::size_t cell = 0; cell < dofmap.extent(0); ++cell)
  {
    auto dofs = stdex::submdspan(dofmap, cell, stdex::full_extent);
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      if (original_to_contiguous[dofs[i]] == -1)
      {
        const std::pair<std::int8_t, std::int32_t>& e = dof_entity[dofs[i]];
        if (e.second < offset[e.first])
          original_to_contiguous[dofs[i]] = counter_owned++;
        else
          original_to_contiguous[dofs[i]] = counter_unowned++;
      }
    }
  }

  // Check for any -1's remaining in `original_to_contiguous` due to vertices
  // on the process that don't belong to a cell. Determine if the dof is owned
  // or a ghost and map to the ends of the owned and ghost "parts" of the
  // contiguous array respectively.
  for (std::size_t dof = 0; dof < original_to_contiguous.size(); ++dof)
  {
    if (original_to_contiguous[dof] == -1)
    {
      if (auto e = dof_entity[dof]; e.second < offset[e.first])
        original_to_contiguous[dof] = counter_owned++;
      else
        original_to_contiguous[dof] = counter_unowned++;
    }
  }

  if (reorder_fn)
  {
    // Re-order using graph ordering

    // Apply graph reordering to owned dofs
    const std::vector<int> node_remap
        = reorder_owned(dofmap, owned_size, original_to_contiguous, reorder_fn);
    std::transform(original_to_contiguous.begin(), original_to_contiguous.end(),
                   original_to_contiguous.begin(),
                   [&node_remap, owned_size](auto index)
                   { return index < owned_size ? node_remap[index] : index; });
  }

  return {std::move(original_to_contiguous), owned_size};
}
//-----------------------------------------------------------------------------

/// Get global indices for unowned dofs
/// @param [in] topology The mesh topology
/// @param [in] num_owned The number of nodes owned by this process
/// @param [in] process_offset The node offset for this process, i.e.
/// the global index of owned node i is i + process_offset
/// @param [in] global_indices_old The old global index of the old local
/// node i
/// @param [in] old_to_new The old local index to new local index map
/// @param [in] dof_entity The ith entry gives (topological dim, local
/// index) of the mesh entity to which node i (old local index) is
/// associated
/// @returns The (0) global indices for unowned dofs, (1) owner rank of
/// each unowned dof
std::pair<std::vector<std::int64_t>, std::vector<int>> get_global_indices(
    const mesh::Topology& topology, const std::int32_t num_owned,
    const std::int64_t process_offset,
    const std::vector<std::int64_t>& global_indices_old,
    const std::vector<std::int32_t>& old_to_new,
    const std::vector<std::pair<std::int8_t, std::int32_t>>& dof_entity)
{
  assert(dof_entity.size() == global_indices_old.size());

  const int D = topology.dim();

  // Build list of flags for owned mesh entities that are shared, i.e.
  // are a ghost on a neighbor
  std::vector<std::vector<std::int8_t>> shared_entity(D + 1);
  for (std::size_t d = 0; d < shared_entity.size(); ++d)
  {
    auto map = topology.index_map(d);
    if (map)
    {
      shared_entity[d] = std::vector<std::int8_t>(map->size_local(), false);
      const std::vector<std::int32_t> forward_indices = map->shared_indices();
      std::for_each(forward_indices.begin(), forward_indices.end(),
                    [&entities = shared_entity[d]](auto idx)
                    { entities[idx] = true; });
    }
  }

  // Build list of (global old, global new) index pairs for dofs that
  // are ghosted on other processes
  std::vector<std::vector<std::int64_t>> global(D + 1);

  // Loop over all dofs
  for (std::size_t i = 0; i < dof_entity.size(); ++i)
  {
    // Topological dimension of mesh entity that dof is associated with
    const int d = dof_entity[i].first;

    // Index of mesh entity that dof is associated with
    const int entity = dof_entity[i].second;
    if (entity < (int)shared_entity[d].size() and shared_entity[d][entity])
    {
      global[d].push_back(global_indices_old[i]);
      global[d].push_back(old_to_new[i] + process_offset);
    }
  }

  std::vector<int> requests_dim;
  std::vector<MPI_Request> requests(D + 1);
  std::vector<MPI_Comm> comm(D + 1, MPI_COMM_NULL);
  std::vector<std::vector<std::int64_t>> all_dofs_received(D + 1);
  std::vector<std::vector<int>> disp_recv(D + 1);
  for (int d = 0; d <= D; ++d)
  {
    // FIXME: This should check which dimension are needed by the dofmap
    auto map = topology.index_map(d);
    if (map)
    {
      const std::vector<int>& src = map->src();
      const std::vector<int>& dest = map->dest();

      MPI_Dist_graph_create_adjacent(
          map->comm(), src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
          dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm[d]);

      // Number and values to send and receive
      const int num_indices = global[d].size();
      std::vector<int> size_recv;
      size_recv.reserve(1); // ensure data is not a nullptr
      size_recv.resize(src.size());
      MPI_Neighbor_allgather(&num_indices, 1, MPI_INT, size_recv.data(), 1,
                             MPI_INT, comm[d]);

      // Compute displacements for data to receive. Last entry has total
      // number of received items.
      disp_recv[d].resize(src.size() + 1);
      std::partial_sum(size_recv.begin(), size_recv.begin() + src.size(),
                       disp_recv[d].begin() + 1);

      // TODO: use MPI_Ineighbor_alltoallv
      // Send global index of dofs to neighbors
      all_dofs_received[d].resize(disp_recv[d].back());
      MPI_Ineighbor_allgatherv(global[d].data(), global[d].size(), MPI_INT64_T,
                               all_dofs_received[d].data(), size_recv.data(),
                               disp_recv[d].data(), MPI_INT64_T, comm[d],
                               &requests[requests_dim.size()]);
      requests_dim.push_back(d);
    }
  }

  // Build  [local_new - num_owned] -> global old array  broken down by
  // dimension
  std::vector<std::vector<std::int64_t>> local_new_to_global_old(D + 1);
  for (std::size_t i = 0; i < global_indices_old.size(); ++i)
  {
    const int d = dof_entity[i].first;
    const std::int32_t local_new = old_to_new[i] - num_owned;
    if (local_new >= 0)
    {
      local_new_to_global_old[d].push_back(global_indices_old[i]);
      local_new_to_global_old[d].push_back(local_new);
    }
  }

  std::vector<std::int64_t> local_to_global_new(old_to_new.size() - num_owned);
  std::vector<int> local_to_global_new_owner(old_to_new.size() - num_owned);
  for (std::size_t i = 0; i < requests_dim.size(); ++i)
  {
    int idx, d;
    MPI_Waitany(requests_dim.size(), requests.data(), &idx, MPI_STATUS_IGNORE);
    d = requests_dim[idx];

    const std::vector<int>& src = topology.index_map(d)->src();

    // Build (global old, global new) map for dofs of dimension d
    std::vector<std::pair<std::int64_t, std::pair<int64_t, int>>>
        global_old_new;
    global_old_new.reserve(disp_recv[d].back());
    for (std::size_t j = 0; j < all_dofs_received[d].size(); j += 2)
    {
      const auto pos
          = std::upper_bound(disp_recv[d].begin(), disp_recv[d].end(), j);
      const int owner = std::distance(disp_recv[d].begin(), pos) - 1;
      global_old_new.push_back(
          {all_dofs_received[d][j], {all_dofs_received[d][j + 1], src[owner]}});
    }
    std::sort(global_old_new.begin(), global_old_new.end());

    // Build the dimension d part of local_to_global_new vector
    for (std::size_t i = 0; i < local_new_to_global_old[d].size(); i += 2)
    {
      std::pair<std::int64_t, std::pair<int64_t, int>> idx_old
          = {local_new_to_global_old[d][i], {0, 0}};

      auto it = std::lower_bound(
          global_old_new.begin(), global_old_new.end(), idx_old,
          [](auto& a, auto& b) { return a.first < b.first; });
      assert(it != global_old_new.end() and it->first == idx_old.first);

      local_to_global_new[local_new_to_global_old[d][i + 1]] = it->second.first;
      local_to_global_new_owner[local_new_to_global_old[d][i + 1]]
          = it->second.second;
    }
  }

  for (std::size_t i = 0; i < comm.size(); ++i)
  {
    if (comm[i] != MPI_COMM_NULL)
      MPI_Comm_free(&comm[i]);
  }

  return {std::move(local_to_global_new), std::move(local_to_global_new_owner)};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<common::IndexMap, int, std::vector<std::int32_t>>
fem::build_dofmap_data(
    MPI_Comm comm, const mesh::Topology& topology,
    const std::vector<ElementDofLayout>& element_dof_layouts,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  common::Timer t0("Build dofmap data");

  const int D = topology.dim();

  // Build a simple dofmap based on mesh entity numbering, returning (i)
  // a local dofmap, (ii) local-to-global map for dof indices, and (iii)
  // pair {dimension, mesh entity index} giving the mesh entity that dof
  // i is associated with.
  const auto [node_graph0, local_to_global0, dof_entity0]
      = build_basic_dofmap(topology, element_dof_layouts);

  // Compute global dofmap offset
  std::int64_t offset = 0;
  for (int d = 0; d <= D; ++d)
  {
    if (element_dof_layouts[0].num_entity_dofs(d) > 0)
    {
      assert(topology.index_map(d));
      offset += topology.index_map(d)->local_range()[0]
                * element_dof_layouts[0].num_entity_dofs(d);
    }
  }

  // Build re-ordering map for data locality and get number of owned
  // nodes
  mdspan2_t<const std::int32_t> _node_graph0(
      node_graph0.data(),
      node_graph0.size() / element_dof_layouts[0].num_dofs(),
      element_dof_layouts[0].num_dofs());
  const auto [old_to_new, num_owned]
      = compute_reordering_map(_node_graph0, dof_entity0, topology, reorder_fn);

  // Get global indices for unowned dofs
  const auto [local_to_global_unowned, local_to_global_owner]
      = get_global_indices(topology, num_owned, offset, local_to_global0,
                           old_to_new, dof_entity0);
  assert(local_to_global_unowned.size() == local_to_global_owner.size());

  // Create IndexMap for dofs range on this process
  common::IndexMap index_map(comm, num_owned, local_to_global_unowned,
                             local_to_global_owner);

  // Build re-ordered dofmap
  std::vector<std::int32_t> dofmap(node_graph0.size());
  for (std::size_t cell = 0; cell < _node_graph0.extent(0); ++cell)
  {
    // Get dof order on this cell
    auto old_nodes = stdex::submdspan(_node_graph0, cell, stdex::full_extent);
    const std::int32_t local_dim0 = old_nodes.size();
    std::span<std::int32_t> dofs(dofmap.data() + cell * local_dim0, local_dim0);
    for (std::int32_t j = 0; j < local_dim0; ++j)
    {
      std::int32_t old_node = old_nodes[j];
      dofs[j] = old_to_new[old_node];
    }
  }

  return {std::move(index_map), element_dof_layouts[0].block_size(),
          std::move(dofmap)};
}
//-----------------------------------------------------------------------------
