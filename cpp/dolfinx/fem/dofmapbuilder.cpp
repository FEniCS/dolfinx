// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dofmapbuilder.h"
#include "ElementDofLayout.h"
#include <algorithm>
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
/// @brief dofmap as a flattened 2D array
/// The number of cells is array.size() / width
/// and therefore array.size() must be a multiple of width
struct dofmap_t
{
  std::int32_t width;
  std::vector<std::int32_t> array;
};
//-----------------------------------------------------------------------------

/// Build a graph for owned dofs and apply graph reordering function with
/// multiple dofmaps. The dofmaps are 2D arrays, of fixed width, stored in
/// `dofmap_t` format. The dofmaps all refer to dof indices in the same range
/// [0:owned_size).
/// @param[in] dofmaps The local dofmaps (cell -> dofs)
/// @param[in] owned_size Number of dofs owned by this process
/// @param[in] original_to_contiguous Map from dof indices in @p dofmap
/// to new indices that are ordered such that owned indices are [0,
/// owned_size)
/// @param[in] reorder_fn The graph reordering function to apply
/// @return Map from original_to_contiguous[i] to new index after
/// reordering
std::vector<int>
reorder_owned(const std::vector<dofmap_t>& dofmaps, std::int32_t owned_size,
              const std::vector<int>& original_to_contiguous,
              const std::function<std::vector<int>(
                  const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  std::vector<std::int32_t> graph_data, graph_offsets;

  // Compute maximum number of graph out edges edges per dof
  std::vector<int> num_edges(owned_size);
  for (const auto& dofmap : dofmaps)
  {
    std::size_t num_cells = dofmap.array.size() / dofmap.width;
    std::vector<std::int32_t> node_temp;
    for (std::size_t cell = 0; cell < num_cells; ++cell)
    {
      node_temp.clear();
      for (std::int32_t i = 0; i < dofmap.width; ++i)
      {
        std::int32_t node
            = original_to_contiguous[dofmap.array[cell * dofmap.width + i]];
        if (node < owned_size)
          node_temp.push_back(node);
      }
      for (std::int32_t node : node_temp)
        num_edges[node] += node_temp.size() - 1;
    }
  }

  // Compute adjacency list with duplicate edges
  std::vector<std::int32_t> offsets(num_edges.size() + 1, 0);
  std::partial_sum(num_edges.begin(), num_edges.end(),
                   std::next(offsets.begin(), 1));
  std::vector<std::int32_t> edges(offsets.back());
  for (const auto& dofmap : dofmaps)
  {
    std::size_t num_cells = dofmap.array.size() / dofmap.width;
    std::vector<std::int32_t> node_temp;

    for (std::size_t cell = 0; cell < num_cells; ++cell)
    {
      node_temp.clear();
      for (std::int32_t i = 0; i < dofmap.width; ++i)
      {
        std::int32_t node
            = original_to_contiguous[dofmap.array[cell * dofmap.width + i]];
        if (node < owned_size)
          node_temp.push_back(node);
      }

      for (std::size_t i = 0; i < node_temp.size(); ++i)
      {
        std::int32_t node_0 = node_temp[i];
        for (std::size_t j = i + 1; j < node_temp.size(); ++j)
        {
          std::int32_t node_1 = node_temp[j];
          edges[offsets[node_0]++] = node_1;
          edges[offsets[node_1]++] = node_0;
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
/// @param [in] element_dof_layout The layout of dofs on each cell type
/// @return Returns: * dofmaps for each cell type (local to the process)
///                  * local-to-global map for each local dof
///                  * local-to-entity map for each local dof
///                  * index maps for each entity type in dofmaps
///                  * the global process offset for dofs on this process
/// @note Entities in the local-to-entity map are represented by the pair:
/// [index_map number, mesh entity index].
std::tuple<std::vector<dofmap_t>, std::vector<std::int64_t>,
           std::vector<std::pair<std::int8_t, std::int32_t>>,
           std::vector<std::shared_ptr<const common::IndexMap>>, std::int64_t>
build_basic_dofmaps(
    const mesh::Topology& topology,
    const std::vector<fem::ElementDofLayout>& element_dof_layouts)
{
  // Start timer for dofmap initialization
  common::Timer t0("Init dofmap from element dofmap");

  // Topological dimension
  const std::size_t D = topology.dim();
  const std::size_t num_cell_types = topology.entity_types(D).size();

  spdlog::info("Checking required entities per dimension");

  // Find which dimensions (d) and entity types (et) are required
  // and the number of dofs which are required for each (d, et) combination.
  // Also store the IndexMaps and local offsets for each.
  std::vector<std::pair<std::int8_t, std::int8_t>> required_dim_et;
  std::vector<std::int32_t> num_entity_dofs_et;
  std::vector<std::shared_ptr<const common::IndexMap>> topo_index_maps;
  std::vector<std::int32_t> local_entity_offsets = {0};

  std::vector<std::vector<mesh::CellType>> entity_types(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    entity_types[d] = topology.entity_types(d);

  for (std::size_t i = 0; i < num_cell_types; ++i)
  {
    mesh::CellType cell_type = entity_types[D][i];
    const std::vector<std::vector<std::vector<int>>>& entity_dofs
        = element_dof_layouts[i].entity_dofs_all();

    for (std::size_t d = 0; d <= D; ++d)
    {
      const std::vector<std::vector<int>>& entity_dofs_d = entity_dofs[d];
      for (std::size_t e = 0; e < entity_dofs_d.size(); ++e)
      {
        if (!entity_dofs_d[e].empty())
        {
          // There is a dof on this entity... find entity type index
          auto et_it = std::find(entity_types[d].begin(), entity_types[d].end(),
                                 mesh::cell_entity_type(cell_type, d, e));
          assert(et_it != entity_types[d].end());
          int et_index = std::distance(entity_types[d].begin(), et_it);

          auto required_entity_it
              = std::find(required_dim_et.begin(), required_dim_et.end(),
                          std::pair<std::int8_t, std::int8_t>{d, et_index});
          if (required_entity_it == required_dim_et.end())
          {
            // Save information for this (d, et) combination
            required_dim_et.push_back({d, et_index});
            const std::int32_t num_entity_dofs = entity_dofs_d[e].size();
            num_entity_dofs_et.push_back(num_entity_dofs);
            auto im = topology.index_maps(d)[et_index];
            topo_index_maps.push_back(im);
            local_entity_offsets.push_back(
                local_entity_offsets.back()
                + num_entity_dofs * (im->size_local() + im->num_ghosts()));

            if (d < D and !topology.connectivity({D, i}, {d, et_index}))
            {
              throw std::runtime_error("Missing needed connectivity. Cell type:"
                                       + std::to_string(i)
                                       + "to dim:" + std::to_string(d)
                                       + ", ent:" + std::to_string(et_index));
            }
          }
          else
          {
            std::size_t k
                = std::distance(required_dim_et.begin(), required_entity_it);
            if (num_entity_dofs_et[k] != (int)entity_dofs_d[e].size())
              throw std::runtime_error("Incompatible elements detected.");
          }
        }
      }
    }
  }

#ifndef NDEBUG
  {
    // Debug output
    std::stringstream s;
    s << "Required entities:";
    for (std::size_t i = 0; i < required_dim_et.size(); ++i)
    {
      s << "(" << (int)required_dim_et[i].first << ", "
        << (int)required_dim_et[i].second << ")=" << num_entity_dofs_et[i]
        << " ";
    }
    spdlog::info("{}", s.str());
  }
#endif

  // Dofmaps on each cell type as (width, [cell_dofs])
  std::vector<dofmap_t> dofs(num_cell_types);
  for (std::size_t i = 0; i < num_cell_types; ++i)
  {
    mesh::CellType cell_type = topology.entity_types(D)[i];
    const std::vector<std::vector<std::vector<int>>> entity_dofs
        = element_dof_layouts[i].entity_dofs_all();
    assert(entity_dofs.size() == D + 1);

    // Loop over cells of this type, and build dofmap
    std::int32_t num_cells = topology.index_maps(D)[i]->size_local()
                             + topology.index_maps(D)[i]->num_ghosts();
    std::int32_t dofmap_width = element_dof_layouts[i].num_dofs();
    dofs[i].width = dofmap_width;
    dofs[i].array.resize(num_cells * dofmap_width);
    spdlog::info("Cell type: {} dofmap: {}x{}", i, num_cells, dofmap_width);

    std::int32_t dofmap_offset = 0;
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      // Wrap dofs for cell c
      std::span<std::int32_t> dofs_c(dofs[i].array.data() + dofmap_offset,
                                     dofmap_width);
      dofmap_offset += dofmap_width;

      // Iterate over required entities for this element, dimension and type
      for (std::size_t k = 0; k < required_dim_et.size(); ++k)
      {
        // Get dimension d and entity type et
        std::size_t d = required_dim_et[k].first;
        std::size_t et = required_dim_et[k].second;
        mesh::CellType e_type = topology.entity_types(d)[et];

        const std::vector<std::vector<int>>& e_dofs_d = entity_dofs[d];

        // Iterate over each entity of current dimension d and type et
        std::span<const std::int32_t> c_to_e
            = d < D ? topology.connectivity({D, i}, {d, et})->links(c)
                    : std::span<const std::int32_t>(&c, 1);

        int w = 0;
        for (std::size_t e = 0; e < e_dofs_d.size(); ++e)
        {
          // Skip entities of wrong type (e.g. for facets of prism)
          // Use separate connectivity index 'w' which only advances for
          // correct entities
          if (mesh::cell_entity_type(cell_type, d, e) == e_type)
          {
            const std::vector<int>& e_dofs_d_e = e_dofs_d[e];
            std::size_t num_entity_dofs = e_dofs_d_e.size();
            assert((int)num_entity_dofs == num_entity_dofs_et[k]);
            std::int32_t e_index_local = c_to_e[w];
            ++w;

            // Loop over dofs belonging to entity e of dimension d (d, e)
            // d: topological dimension
            // e: local entity index
            // dof_local: local index of dof at (d, e)
            for (std::size_t j = 0; j < num_entity_dofs; ++j)
            {
              int dof_local = e_dofs_d_e[j];
              dofs_c[dof_local] = local_entity_offsets[k]
                                  + num_entity_dofs * e_index_local + j;
            }
          }
        }
      }
    }
  }

  spdlog::info("Global index computation");

  // TODO: Put Global index computations in separate function
  // Global index computations

  // Create local to global map and dof entity map.
  // NOTE: this must be done outside of the above loop as some processes
  // may have vertices that don't belong to a cell on that process.

  // Dof -> (index_map number, entity index) marker
  std::int32_t local_size = local_entity_offsets.back();
  std::vector<std::pair<std::int8_t, std::int32_t>> dof_entity(local_size);

  // Storage for local-to-global map
  std::vector<std::int64_t> local_to_global(local_size);

  std::int64_t global_entity_offsets = 0;
  std::int64_t global_start = 0;
  for (std::size_t k = 0; k < required_dim_et.size(); ++k)
  {
    const int num_entity_dofs = num_entity_dofs_et[k];
    auto map = topo_index_maps[k];
    assert(map);
    std::vector<std::int64_t> global_indices = map->global_indices();

    for (std::size_t e_index = 0; e_index < global_indices.size(); ++e_index)
    {
      auto e_index_global = global_indices[e_index];
      for (std::int32_t count = 0; count < num_entity_dofs; ++count)
      {
        std::int32_t dof
            = local_entity_offsets[k] + num_entity_dofs * e_index + count;
        local_to_global[dof]
            = global_entity_offsets + num_entity_dofs * e_index_global + count;
        dof_entity[dof] = {k, e_index};
      }
    }
    global_entity_offsets += num_entity_dofs * map->size_global();
    global_start += num_entity_dofs * map->local_range()[0];
  }

  return {std::move(dofs), std::move(local_to_global), std::move(dof_entity),
          std::move(topo_index_maps), global_start};
}
//-----------------------------------------------------------------------------

/// Compute re-ordering map from old local index to new local index. The
/// M dofs owned by this process are reordered for locality and fill the
/// positions [0, ..., M). Dof owned by another process are placed at
/// the end, i.e. in the positions [M, ..., N), where N is the total
/// number of dofs on this process.
///
/// @param [in] dofmaps The basic dofmap data in multiple dofmaps sharing the
/// same range
/// @param [in] dof_entity Map from dof index to (index_map, entity_index),
/// where entity_index is the local mesh entity index in the given index_map
/// @param [in] index_maps The set of IndexMaps, one for each topological
/// entity type used in the dofmap. The location in this array is referred to by
/// the first item in each entry of @p dof_entity
/// @param [in] reorder_fn Graph reordering function that is applied for
/// dof re-ordering
/// @return The pair (old-to-new local index map, M), where M is the
/// number of dofs owned by this process
std::pair<std::vector<std::int32_t>, std::int32_t> compute_reordering_map(
    const std::vector<dofmap_t>& dofmaps,
    const std::vector<std::pair<std::int8_t, std::int32_t>>& dof_entity,
    const std::vector<std::shared_ptr<const common::IndexMap>>& index_maps,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  common::Timer t0("Compute dof reordering map");

  // Get mesh entity ownership offset for each IndexMap
  std::vector<std::int32_t> offset(index_maps.size(), -1);
  std::transform(index_maps.begin(), index_maps.end(), offset.begin(),
                 [](auto map) { return map->size_local(); });

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
  for (auto dofmap : dofmaps)
  {
    for (std::int32_t dof : dofmap.array)
    {
      if (original_to_contiguous[dof] == -1)
      {
        if (auto e = dof_entity[dof]; e.second < offset[e.first])
          original_to_contiguous[dof] = counter_owned++;
        else
          original_to_contiguous[dof] = counter_unowned++;
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
    const std::vector<int> node_remap = reorder_owned(
        dofmaps, owned_size, original_to_contiguous, reorder_fn);
    std::transform(original_to_contiguous.begin(), original_to_contiguous.end(),
                   original_to_contiguous.begin(),
                   [&node_remap, owned_size](auto index)
                   { return index < owned_size ? node_remap[index] : index; });
  }

  return {std::move(original_to_contiguous), owned_size};
}
//-----------------------------------------------------------------------------

/// Get global indices for unowned dofs
/// @param [in] index_maps Set of index maps corresponding to dofs in @p
/// dof_entity, below.
/// @param [in] num_owned The number of nodes owned by this process
/// @param [in] process_offset The node offset for this process, i.e.
/// the global index of owned node i is i + process_offset
/// @param [in] global_indices_old The old global index of the old local
/// node i
/// @param [in] old_to_new The old local index to new local index map
/// @param [in] dof_entity The ith entry gives (index_map, local
/// index) of the mesh entity to which node i (old local index) is
/// associated.
/// @returns The (0) global indices for unowned dofs, (1) owner rank of
/// each unowned dof
std::pair<std::vector<std::int64_t>, std::vector<int>> get_global_indices(
    const std::vector<std::shared_ptr<const common::IndexMap>>& index_maps,
    std::int32_t num_owned, std::int64_t process_offset,
    const std::vector<std::int64_t>& global_indices_old,
    const std::vector<std::int32_t>& old_to_new,
    const std::vector<std::pair<std::int8_t, std::int32_t>>& dof_entity)
{
  assert(dof_entity.size() == global_indices_old.size());

  // Build list of flags for owned mesh entities that are shared, i.e.
  // are a ghost on a neighbor
  std::vector<std::vector<std::int8_t>> shared_entity(index_maps.size());
  for (std::size_t d = 0; d < index_maps.size(); ++d)
  {
    auto map = index_maps[d];
    assert(map);

    shared_entity[d] = std::vector<std::int8_t>(map->size_local(), false);
    const std::vector<std::int32_t> forward_indices = map->shared_indices();
    std::for_each(forward_indices.begin(), forward_indices.end(),
                  [&entities = shared_entity[d]](auto idx)
                  { entities[idx] = true; });
  }

  // Build list of (global old, global new) index pairs for dofs that
  // are ghosted on other processes

  // Loop over all dofs
  std::vector<std::vector<std::int64_t>> global(index_maps.size());
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
  std::vector<MPI_Request> requests(index_maps.size());
  std::vector<MPI_Comm> comm(index_maps.size(), MPI_COMM_NULL);
  std::vector<std::vector<std::int64_t>> all_dofs_received(index_maps.size());
  std::vector<std::vector<int>> disp_recv(index_maps.size());
  for (std::size_t d = 0; d < index_maps.size(); ++d)
  {
    auto map = index_maps[d];
    assert(map);

    std::span src = map->src();
    std::span dest = map->dest();
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

  // Build  [local_new - num_owned] -> global old array  broken down by
  // dimension
  std::vector<std::vector<std::int64_t>> local_new_to_global_old(
      index_maps.size());
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

    std::span src = index_maps[d]->src();

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
    std::ranges::sort(global_old_new);

    // Build the dimension d part of local_to_global_new vector
    for (std::size_t i = 0; i < local_new_to_global_old[d].size(); i += 2)
    {
      std::pair<std::int64_t, std::pair<int64_t, int>> idx_old
          = {local_new_to_global_old[d][i], {0, 0}};

      auto it = std::lower_bound(global_old_new.begin(), global_old_new.end(),
                                 idx_old, [](auto& a, auto& b)
                                 { return a.first < b.first; });
      assert(it != global_old_new.end() and it->first == idx_old.first);

      local_to_global_new[local_new_to_global_old[d][i + 1]] = it->second.first;
      local_to_global_new_owner[local_new_to_global_old[d][i + 1]]
          = it->second.second;
    }
  }

  for (MPI_Comm c : comm)
  {
    if (c != MPI_COMM_NULL)
      MPI_Comm_free(&c);
  }

  return {std::move(local_to_global_new), std::move(local_to_global_new_owner)};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<common::IndexMap, int, std::vector<std::vector<std::int32_t>>>
fem::build_dofmap_data(
    MPI_Comm comm, const mesh::Topology& topology,
    const std::vector<ElementDofLayout>& element_dof_layouts,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  common::Timer t0("Build dofmap data");

  // Build a simple dofmap based on mesh entity numbering, returning (i)
  // a local dofmap, (ii) local-to-global map for dof indices, and (iii)
  // pair {dimension, mesh entity index} giving the mesh entity that dof
  // i is associated with.
  const auto [node_graphs, local_to_global0, dof_entity0, topo_index_maps,
              offset]
      = build_basic_dofmaps(topology, element_dof_layouts);

  spdlog::info("Got {} index_maps", topo_index_maps.size());

  // Build re-ordering map for data locality and get number of owned
  // nodes
  const auto [old_to_new, num_owned] = compute_reordering_map(
      node_graphs, dof_entity0, topo_index_maps, reorder_fn);

  spdlog::info("Get global indices");

  // Get global indices for unowned dofs
  const auto [local_to_global_unowned, local_to_global_owner]
      = get_global_indices(topo_index_maps, num_owned, offset, local_to_global0,
                           old_to_new, dof_entity0);
  assert(local_to_global_unowned.size() == local_to_global_owner.size());

  // Create IndexMap for dofs range on this process
  common::IndexMap index_map(comm, num_owned, local_to_global_unowned,
                             local_to_global_owner);

  // Build re-ordered dofmaps
  std::vector<std::vector<std::int32_t>> dofmaps(node_graphs.size());
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
  {
    const std::vector<std::int32_t>& node_graphs_i = node_graphs[i].array;
    dofmaps[i].resize(node_graphs_i.size());
    std::vector<std::int32_t>& dofmaps_i = dofmaps[i];
    for (std::size_t j = 0; j < node_graphs_i.size(); ++j)
    {
      std::int32_t old_node = node_graphs_i[j];
      dofmaps_i[j] = old_to_new[old_node];
    }
  }

  return {std::move(index_map), element_dof_layouts.front().block_size(),
          std::move(dofmaps)};
}
//-----------------------------------------------------------------------------
