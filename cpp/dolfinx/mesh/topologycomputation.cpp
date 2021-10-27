// Copyright (C) 2006-2020 Anders Logg, Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "topologycomputation.h"
#include "Topology.h"
#include "cell_types.h"
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
/// Get the ownership of an entity shared over several processes
/// @param processes Set of sharing processes
/// @param vertices Global vertex indices of entity
/// @return owning process number
int get_ownership(std::set<int>& processes, std::vector<std::int64_t>& vertices)
{
  // Use a deterministic random number generator, seeded with global vertex
  // indices ensuring all processes get the same answer
  std::mt19937 gen;
  std::seed_seq seq(vertices.begin(), vertices.end());
  gen.seed(seq);
  std::vector<int> p(processes.begin(), processes.end());
  int index = gen() % p.size();
  int owner = p[index];
  return owner;
}
//-----------------------------------------------------------------------------

/// Communicate with sharing processes to find out which entities are
/// ghosts and return a map (vector) to move these local indices to the
/// end of the local range. Also returns the index map, and shared
/// entities, i.e. the set of all processes which share each shared
/// entity.
/// @param[in] comm MPI Communicator
/// @param[in] cell_indexmap IndexMap for cells
/// @param[in] vertex_indexmap IndexMap for vertices
/// @param[in] entity_list List of entities, each entity represented by
/// its local vertex indices
/// @param[in] num_vertices_per_e Number of vertices per entity
/// @param[in] entity_index Initial numbering for each row in
/// entity_list
/// @returns Tuple of (local_indices, index map, shared entities)
std::tuple<std::vector<int>, common::IndexMap>
get_local_indexing(MPI_Comm comm, const common::IndexMap& cell_indexmap,
                   const common::IndexMap& vertex_indexmap,
                   const xt::xtensor<std::int32_t, 2>& entity_list,
                   const std::vector<std::int32_t>& entity_index)
{
  // entity_list contains all the entities for all the cells,
  // listed as local vertex indices, and entity_index contains the
  // initial numbering of the entities.
  //              entity_list entity_index
  // e.g. cell0-ent0: [0,1,2] 15
  //      cell0-ent1: [1,2,3] 23
  //      cell1-ent0: [0,1,2] 15
  //      cell1-ent1: [1,2,6] 24
  //      ...

  // Find the maximum entity index, hence the number of entities
  std::int32_t entity_count = 0;
  if (auto mx = std::max_element(entity_index.begin(), entity_index.end());
      mx != entity_index.end())
  {
    entity_count = *mx + 1;
  }

  //---------
  // Set ghost status array values
  // 1 = entities that are only in local cells (i.e. owned)
  // 2 = entities that are only in ghost cells (i.e. not owned)
  // 3 = entities with ownership that needs deciding (used also for unghosted
  // case)
  std::vector<int> ghost_status(entity_count, 0);
  {
    if (cell_indexmap.num_ghosts() == 0)
      std::fill(ghost_status.begin(), ghost_status.end(), 3);
    else
    {
      const std::int32_t num_cells
          = cell_indexmap.size_local() + cell_indexmap.num_ghosts();
      assert(entity_list.shape(0) % num_cells == 0);
      const std::int32_t num_entities_per_cell
          = entity_list.shape(0) / num_cells;
      const std::int32_t ghost_offset
          = cell_indexmap.size_local() * num_entities_per_cell;

      // Tag all entities in local cells with 1
      for (int i = 0; i < ghost_offset; ++i)
      {
        const std::int32_t idx = entity_index[i];
        ghost_status[idx] = 1;
      }

      // Set entities in ghost cells to 2 (purely ghost) or 3 (border)
      for (std::size_t i = ghost_offset; i < entity_list.shape(0); ++i)
      {
        const std::int32_t idx = entity_index[i];
        ghost_status[idx] = ghost_status[idx] | 2;
      }
    }
  }

  //---------
  // Create an expanded neighbor_comm from shared_vertices
  const std::map<std::int32_t, std::set<std::int32_t>> shared_vertices
      = vertex_indexmap.compute_shared_indices();

  std::set<std::int32_t> neighbor_set;
  for (auto& q : shared_vertices)
    neighbor_set.insert(q.second.begin(), q.second.end());
  std::vector<std::int32_t> neighbors(neighbor_set.begin(), neighbor_set.end());
  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbor_comm);

  const int neighbor_size = neighbors.size();
  std::unordered_map<int, int> proc_to_neighbor;
  for (int i = 0; i < neighbor_size; ++i)
    proc_to_neighbor.insert({neighbors[i], i});

  std::vector<std::vector<std::int64_t>> send_entities(neighbor_size);
  std::vector<std::vector<std::int32_t>> send_index(neighbor_size);

  // Get all "possibly shared" entities, based on vertex sharing. Send
  // to other processes, and see if we get the same back.

  // Map for entities to entity index, using global vertex indices
  std::map<std::vector<std::int64_t>, std::int32_t>
      global_entity_to_entity_index;

  // Set of sharing procs for each entity, counting vertex hits
  // Get a list of unique entities
  std::vector<std::int32_t> unique_row(entity_count);
  for (std::size_t i = 0; i < entity_list.shape(0); ++i)
    unique_row[entity_index[i]] = i;
  const int num_vertices_per_e = entity_list.shape(1);
  std::unordered_map<int, int> procs;
  std::vector<std::int64_t> vglobal(num_vertices_per_e);
  std::vector<std::int32_t> entity_list_i(num_vertices_per_e);
  for (int i : unique_row)
  {
    std::copy_n(xt::row(entity_list, i).begin(), num_vertices_per_e,
                entity_list_i.begin());
    procs.clear();
    for (int j = 0; j < num_vertices_per_e; ++j)
    {
      if (auto it = shared_vertices.find(entity_list_i[j]);
          it != shared_vertices.end())
      {
        for (std::int32_t p : it->second)
          ++procs[p];
      }
    }

    for (const auto& q : procs)
    {
      // If any process shares all vertices, then add to list
      if (q.second == num_vertices_per_e)
      {
        vertex_indexmap.local_to_global(entity_list_i, vglobal);
        dolfinx::radix_sort(xtl::span(vglobal));

        global_entity_to_entity_index.insert({vglobal, entity_index[i]});

        // Do not send entities which are known to be ghosts
        if (ghost_status[entity_index[i]] != 2)
        {
          const int p = q.first;
          auto it = proc_to_neighbor.find(p);
          assert(it != proc_to_neighbor.end());
          const int np = it->second;

          // Entity entity_index[i] may be shared with process p
          send_entities[np].insert(send_entities[np].end(), vglobal.begin(),
                                   vglobal.end());
          send_index[np].push_back(entity_index[i]);
        }
      }
    }
  }

  // Get shared entities of this dimension, and also match up an index
  // for the received entities (from other processes) with the indices
  // of the sent entities (to other processes)

  const graph::AdjacencyList<std::int64_t> recv_data
      = dolfinx::MPI::neighbor_all_to_all(
          neighbor_comm, graph::AdjacencyList<std::int64_t>(send_entities));

  const std::vector<std::int64_t>& recv_entities_data = recv_data.array();
  const std::vector<std::int32_t>& recv_offsets = recv_data.offsets();

  // Compare received with sent for each process
  // Any which are not found will have -1 in recv_index
  std::vector<std::int32_t> recv_index;
  std::vector<std::int64_t> recv_vec;
  std::unordered_map<std::int32_t, std::vector<std::int64_t>>
      shared_entity_to_global_vertices;
  std::unordered_map<std::int32_t, std::set<std::int32_t>> shared_entities;
  for (int np = 0; np < neighbor_size; ++np)
  {
    for (int j = recv_offsets[np]; j < recv_offsets[np + 1];
         j += num_vertices_per_e)
    {
      recv_vec.assign(
          std::next(recv_entities_data.begin(), j),
          std::next(recv_entities_data.begin(), j + num_vertices_per_e));
      if (auto it = global_entity_to_entity_index.find(recv_vec);
          it != global_entity_to_entity_index.end())
      {
        const int p = neighbors[np];
        shared_entities[it->second].insert(p);
        shared_entity_to_global_vertices.insert({it->second, recv_vec});
        recv_index.push_back(it->second);
      }
      else
        recv_index.push_back(-1);
    }
  }

  // Add this rank to the list of sharing processes
  const int mpi_rank = dolfinx::MPI::rank(comm);
  std::for_each(shared_entities.begin(), shared_entities.end(),
                [mpi_rank](auto& q) { q.second.insert(mpi_rank); });

  //---------
  // Determine ownership
  std::vector<std::int32_t> local_index(entity_count, -1);
  std::int32_t num_local;
  {
    std::int32_t c = 0;
    // Index non-ghost entities
    for (int i = 0; i < entity_count; ++i)
    {
      assert(ghost_status[i] > 0);
      // Definitely ghost
      if (ghost_status[i] == 2)
        continue;

      // Definitely local
      if (const auto it = shared_entities.find(i);
          ghost_status[i] == 1 or it == shared_entities.end())
      {
        local_index[i] = c;
        ++c;
      }
      else
      {
        const auto global_vertices_it
            = shared_entity_to_global_vertices.find(i);
        assert(global_vertices_it != shared_entity_to_global_vertices.end());
        int owner_rank = get_ownership(it->second, global_vertices_it->second);
        if (owner_rank == mpi_rank)
        {
          // Take ownership
          local_index[i] = c;
          ++c;
        }
      }
    }
    num_local = c;

    std::transform(local_index.cbegin(), local_index.cend(),
                   local_index.begin(),
                   [&c](auto index) { return index == -1 ? c++ : index; });

    assert(c == entity_count);
  }

  //---------
  // Communicate global indices to other processes
  std::vector<int> ghost_owners(entity_count - num_local, -1);
  std::vector<std::int64_t> ghost_indices(entity_count - num_local, -1);
  {
    const std::int64_t _num_local = num_local;
    std::int64_t local_offset = 0;
    MPI_Exscan(&_num_local, &local_offset, 1,
               dolfinx::MPI::mpi_type<std::int64_t>(), MPI_SUM, comm);

    std::vector<std::int64_t> send_global_index_data;
    std::vector<int> send_global_index_offsets = {0};

    // Send global indices for same entities that we sent before. This
    // uses the same pattern as before, so we can match up the received
    // data to the indices in recv_index
    for (const auto& indices : send_index)
    {
      std::transform(indices.cbegin(), indices.cend(),
                     std::back_inserter(send_global_index_data),
                     [&local_index, num_local,
                      local_offset](std::int32_t index) -> std::int64_t
                     {
                       // If not in our local range, send -1.
                       return local_index[index] < num_local
                                  ? local_offset + local_index[index]
                                  : -1;
                     });
      send_global_index_offsets.push_back(send_global_index_data.size());
    }

    const graph::AdjacencyList<std::int64_t> recv_data
        = dolfinx::MPI::neighbor_all_to_all(
            neighbor_comm,
            graph::AdjacencyList<std::int64_t>(send_global_index_data,
                                               send_global_index_offsets));

    const std::vector<std::int64_t>& recv_global_index_data = recv_data.array();
    const std::vector<std::int32_t>& recv_offsets = recv_data.offsets();
    assert(recv_global_index_data.size() == recv_index.size());

    // Map back received indices
    for (std::size_t j = 0; j < recv_global_index_data.size(); ++j)
    {
      const std::int64_t gi = recv_global_index_data[j];
      const std::int32_t idx = recv_index[j];
      if (gi != -1 and idx != -1)
      {
        assert(local_index[idx] >= num_local);
        ghost_indices[local_index[idx] - num_local] = gi;
        auto pos
            = std::upper_bound(recv_offsets.begin(), recv_offsets.end(), j);
        const int owner = std::distance(recv_offsets.begin(), pos) - 1;
        ghost_owners[local_index[idx] - num_local] = neighbors[owner];
      }
    }

    assert(std::find(ghost_indices.begin(), ghost_indices.end(), -1)
           == ghost_indices.end());
  }

  MPI_Comm_free(&neighbor_comm);

  common::IndexMap index_map(
      comm, num_local,
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_owners.begin(), ghost_owners.end())),
      ghost_indices, ghost_owners);

  // Map from initial numbering to new local indices
  std::vector<std::int32_t> new_entity_index(entity_index.size());
  std::transform(entity_index.cbegin(), entity_index.cend(),
                 new_entity_index.begin(),
                 [&local_index](auto index) { return local_index[index]; });

  return {std::move(new_entity_index), std::move(index_map)};
}
//-----------------------------------------------------------------------------

/// Compute entities of dimension d
/// @param[in] comm MPI communicator (TODO: full or neighbor hood?)
/// @param[in] cells Adjacency list for cell-vertex connectivity
/// @param[in] shared_vertices TODO
/// @param[in] cell_type Cell type
/// @param[in] dim Topological dimension of the entities to be computed
/// @return Returns the (cell-entity connectivity, entity-cell
///   connectivity, index map for the entity distribution across
///   processes, shared entities)
std::tuple<graph::AdjacencyList<std::int32_t>,
           graph::AdjacencyList<std::int32_t>, common::IndexMap>
compute_entities_by_key_matching(
    MPI_Comm comm, const graph::AdjacencyList<std::int32_t>& cells,
    const common::IndexMap& vertex_index_map,
    const common::IndexMap& cell_index_map, mesh::CellType cell_type, int dim)
{
  if (dim == 0)
  {
    throw std::runtime_error(
        "Cannot create vertices for topology. Should already exist.");
  }

  // Start timer
  common::Timer timer("Compute entities of dim = " + std::to_string(dim));

  // Initialize local array of entities
  const std::int8_t num_entities_per_cell
      = mesh::cell_num_entities(cell_type, dim);

  // For some cells, the num_vertices varies per facet (3 or 4)
  int max_vertices_per_entity = 0;
  for (int i = 0; i < num_entities_per_cell; ++i)
  {
    max_vertices_per_entity = std::max(
        max_vertices_per_entity,
        mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, dim, i)));
  }

  // Create map from cell vertices to entity vertices
  auto e_vertices = mesh::get_entity_vertices(cell_type, dim);

  // List of vertices for each entity in each cell
  const std::size_t num_cells = cells.num_nodes();
  xt::xtensor<std::int32_t, 2> entity_list(
      {num_cells * num_entities_per_cell,
       (std::size_t)max_vertices_per_entity});
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // Get vertices from cell
    auto vertices = cells.links(c);

    for (int i = 0; i < num_entities_per_cell; ++i)
    {
      const std::int32_t idx = c * num_entities_per_cell + i;
      auto ev = e_vertices.links(i);

      // Get entity vertices padding with -1 if fewer than
      // max_vertices_per_entity
      entity_list(idx, max_vertices_per_entity - 1) = -1;
      for (std::size_t j = 0; j < ev.size(); ++j)
        entity_list(idx, j) = vertices[ev[j]];
    }
  }

  // Copy list and sort vertices of each entity into (reverse) order
  xt::xtensor<std::int32_t, 2> entity_list_sorted = entity_list;
  for (std::size_t i = 0; i < entity_list_sorted.shape(0); ++i)
  {
    std::sort(xt::row(entity_list_sorted, i).begin(),
              xt::row(entity_list_sorted, i).end(), std::greater<>());
  }

  // Sort the list and label uniquely
  const std::vector<std::int32_t> sort_order
      = dolfinx::sort_by_perm(entity_list_sorted);

  std::vector<std::int32_t> entity_index(entity_list.shape(0), 0);
  std::int32_t entity_count = 0;
  std::int32_t last = sort_order[0];
  for (std::size_t i = 1; i < sort_order.size(); ++i)
  {
    std::int32_t j = sort_order[i];
    if (xt::row(entity_list_sorted, j) != xt::row(entity_list_sorted, last))
      ++entity_count;
    entity_index[j] = entity_count;
    last = j;
  }
  ++entity_count;

  // Communicate with other processes to find out which entities are
  // ghosted and shared. Remap the numbering so that ghosts are at the
  // end.
  auto [local_index, index_map] = get_local_indexing(
      comm, cell_index_map, vertex_index_map, entity_list, entity_index);

  // Entity-vertex connectivity
  std::vector<std::int32_t> offsets_ev(entity_count + 1, 0);
  std::vector<int> size_ev(entity_count);
  for (std::size_t i = 0; i < entity_list.shape(0); ++i)
  {
    size_ev[local_index[i]]
        = (entity_list(i, max_vertices_per_entity - 1) == -1)
              ? (max_vertices_per_entity - 1)
              : max_vertices_per_entity;
  }
  for (int i = 0; i < entity_count; ++i)
    offsets_ev[i + 1] = offsets_ev[i] + size_ev[i];

  graph::AdjacencyList<std::int32_t> ev(
      std::vector<std::int32_t>(offsets_ev.back()), std::move(offsets_ev));
  for (std::size_t i = 0; i < entity_list.shape(0); ++i)
  {
    std::copy_n(xt::row(entity_list, i).begin(), ev.num_links(local_index[i]),
                ev.links(local_index[i]).begin());
  }

  // NOTE: Cell-entity connectivity comes after ev creation because
  // below we use std::move(local_index)

  // Cell-entity connectivity
  std::vector<std::int32_t> offsets_ce(num_cells + 1, 0);
  for (std::size_t i = 0; i < offsets_ce.size() - 1; ++i)
    offsets_ce[i + 1] = offsets_ce[i] + num_entities_per_cell;
  graph::AdjacencyList<std::int32_t> ce(std::move(local_index),
                                        std::move(offsets_ce));

  return {std::move(ce), std::move(ev), std::move(index_map)};
}
//-----------------------------------------------------------------------------

/// Compute connectivity from entities of dimension d0 to entities of
/// dimension d1 using the transpose connectivity (d1 -> d0)
/// @param[in] c_d1_d0 The connectivity from entities of dimension d1 to
///   entities of dimension d0
/// @param[in] num_entities_d0 The number of entities of dimension d0
/// @return The connectivity from entities of dimension d0 to entities
///   of dimension d1
graph::AdjacencyList<std::int32_t>
compute_from_transpose(const graph::AdjacencyList<std::int32_t>& c_d1_d0,
                       const int num_entities_d0, int d0, int d1)
{
  LOG(INFO) << "Computing mesh connectivity " << d0 << " - " << d1
            << " from transpose.";

  // Compute number of connections for each e0
  std::vector<std::int32_t> num_connections(num_entities_d0, 0);
  for (int e1 = 0; e1 < c_d1_d0.num_nodes(); ++e1)
  {
    for (std::int32_t e0 : c_d1_d0.links(e1))
      num_connections[e0]++;
  }

  // Compute offsets
  std::vector<std::int32_t> offsets(num_connections.size() + 1, 0);
  std::partial_sum(num_connections.begin(), num_connections.end(),
                   std::next(offsets.begin()));

  std::vector<std::int32_t> counter(num_connections.size(), 0);
  std::vector<std::int32_t> connections(offsets[offsets.size() - 1]);
  for (int e1 = 0; e1 < c_d1_d0.num_nodes(); ++e1)
    for (std::int32_t e0 : c_d1_d0.links(e1))
      connections[offsets[e0] + counter[e0]++] = e1;

  return graph::AdjacencyList<std::int32_t>(std::move(connections),
                                            std::move(offsets));
}
//-----------------------------------------------------------------------------

/// Compute the d0 -> d1 connectivity, where d0 > d1
/// @param[in] c_d0_0 The d0 -> 0 (entity (d0) to vertex) connectivity
/// @param[in] c_d0_0 The d1 -> 0 (entity (d1) to vertex) connectivity
/// @param[in] cell_type_d0 The cell type for entities of dimension d0
/// @param[in] d0 Topological dimension
/// @param[in] d1 Topological dimension
/// @return The d0 -> d1 connectivity
graph::AdjacencyList<std::int32_t>
compute_from_map(const graph::AdjacencyList<std::int32_t>& c_d0_0,
                 const graph::AdjacencyList<std::int32_t>& c_d1_0, int d0,
                 int d1)
{
  // Only possible case is facet->edge
  assert(d0 == 2 and d1 == 1);

  // Make a map from the sorted edge vertices to the edge index
  boost::unordered_map<std::array<std::int32_t, 2>, std::int32_t> edge_to_index;
  edge_to_index.reserve(c_d1_0.num_nodes());

  std::array<std::int32_t, 2> key;
  for (int e = 0; e < c_d1_0.num_nodes(); ++e)
  {
    xtl::span<const std::int32_t> v = c_d1_0.links(e);
    assert(v.size() == key.size());
    std::partial_sort_copy(v.begin(), v.end(), key.begin(), key.end());
    edge_to_index.insert({key, e});
  }

  // Number of edges for a tri/quad is the same as number of vertices
  // so AdjacencyList will have same offset pattern
  std::vector<std::int32_t> connections;
  connections.reserve(c_d0_0.array().size());
  std::vector<std::int32_t> offsets(c_d0_0.offsets());

  // Search for edges of facet in map, and recover index
  const auto tri_vertices_ref
      = mesh::get_entity_vertices(mesh::CellType::triangle, 1);
  const auto quad_vertices_ref
      = mesh::get_entity_vertices(mesh::CellType::quadrilateral, 1);

  for (int e = 0; e < c_d0_0.num_nodes(); ++e)
  {
    auto e0 = c_d0_0.links(e);
    auto vref = (e0.size() == 3) ? &tri_vertices_ref : &quad_vertices_ref;
    for (std::size_t i = 0; i < e0.size(); ++i)
    {
      const auto& v = vref->links(i);
      for (int j = 0; j < 2; ++j)
        key[j] = e0[v[j]];
      std::sort(key.begin(), key.end());
      const auto it = edge_to_index.find(key);
      assert(it != edge_to_index.end());
      connections.push_back(it->second);
    }
  }

  connections.shrink_to_fit();
  return graph::AdjacencyList<std::int32_t>(std::move(connections),
                                            std::move(offsets));
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>>
mesh::compute_entities(MPI_Comm comm, const Topology& topology, int dim)
{
  LOG(INFO) << "Computing mesh entities of dimension " << dim;
  const int tdim = topology.dim();

  // Vertices must always exist
  if (dim == 0)
    return {nullptr, nullptr, nullptr};

  if (topology.connectivity(dim, 0))
  {
    // Make sure we really have the connectivity
    if (!topology.connectivity(tdim, dim))
    {
      throw std::runtime_error(
          "Cannot compute topological entities. Entities of topological "
          "dimension "
          + std::to_string(dim)
          + " exist but cell-dim connectivity is missing.");
    }
    return {nullptr, nullptr, nullptr};
  }

  auto cells = topology.connectivity(tdim, 0);
  if (!cells)
    throw std::runtime_error("Cell connectivity missing.");

  auto vertex_map = topology.index_map(0);
  assert(vertex_map);
  auto cell_map = topology.index_map(tdim);
  assert(cell_map);
  auto [d0, d1, d2] = compute_entities_by_key_matching(
      comm, *cells, *vertex_map, *cell_map, topology.cell_type(), dim);

  return {std::make_shared<graph::AdjacencyList<std::int32_t>>(std::move(d0)),
          std::make_shared<graph::AdjacencyList<std::int32_t>>(std::move(d1)),
          std::make_shared<common::IndexMap>(std::move(d2))};
}
//-----------------------------------------------------------------------------
std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
mesh::compute_connectivity(const Topology& topology, int d0, int d1)
{
  LOG(INFO) << "Requesting connectivity " << d0 << " - " << d1;

  // Return if connectivity has already been computed
  if (topology.connectivity(d0, d1))
    return {nullptr, nullptr};

  // Get entities exist
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d0_0
      = topology.connectivity(d0, 0);
  if (d0 > 0 and !topology.connectivity(d0, 0))
  {
    throw std::runtime_error("Missing entities of dimension "
                             + std::to_string(d0) + ".");
  }

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d1_0
      = topology.connectivity(d1, 0);
  if (d1 > 0 and !topology.connectivity(d1, 0))
  {
    throw std::runtime_error("Missing entities of dimension "
                             + std::to_string(d1) + ".");
  }

  // Start timer
  common::Timer timer("Compute connectivity " + std::to_string(d0) + "-"
                      + std::to_string(d1));

  // Decide how to compute the connectivity
  if (d0 == d1)
  {
    return {std::make_shared<graph::AdjacencyList<std::int32_t>>(
                c_d0_0->num_nodes()),
            nullptr};
  }
  else if (d0 < d1)
  {
    // Compute connectivity d1 - d0 (if needed), and take transpose
    if (!topology.connectivity(d1, d0))
    {
      auto c_d1_d0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_map(*c_d1_0, *c_d0_0, d1, d0));
      auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_transpose(*c_d1_d0, c_d0_0->num_nodes(), d0, d1));
      return {c_d0_d1, c_d1_d0};
    }
    else
    {
      assert(c_d0_0);
      assert(topology.connectivity(d1, d0));
      auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_transpose(*topology.connectivity(d1, d0),
                                 c_d0_0->num_nodes(), d0, d1));
      return {c_d0_d1, nullptr};
    }
  }
  else if (d0 > d1)
  {
    // Compute by mapping vertices from a lower dimension entity to
    // those of a higher dimension entity
    auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        compute_from_map(*c_d0_0, *c_d1_0, d0, d1));
    return {c_d0_d1, nullptr};
  }
  else
    throw std::runtime_error("Entity dimension error when computing topology.");
}
//--------------------------------------------------------------------------
