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
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{

/// @brief Create an adjacency list from array of pairs, where the first
/// value in the pair is the node and the second value is the edge.
/// @param[in] data List if pairs
/// @param[in] size The number of edges in the graph. For example, this
/// can be used to build an adjacency list that includes 'owned' nodes only.
/// @pre The `data` array must be sorted.
template <typename U>
graph::AdjacencyList<int> create_adj_list(U& data, std::int32_t size)
{
  std::sort(data.begin(), data.end());
  data.erase(std::unique(data.begin(), data.end()), data.end());

  std::vector<int> array;
  array.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(array),
                 [](auto x) { return x.second; });

  std::vector<std::int32_t> offsets{0};
  offsets.reserve(size + 1);
  auto it = data.begin();
  for (std::int32_t e = 0; e < size; ++e)
  {
    auto it1
        = std::find_if(it, data.end(), [e](auto x) { return x.first != e; });
    offsets.push_back(offsets.back() + std::distance(it, it1));
    it = it1;
  }

  return graph::AdjacencyList<int>(std::move(array), std::move(offsets));
}

//-----------------------------------------------------------------------------
/// Get the ownership of an entity shared over several processes
/// @param processes Set of sharing processes
/// @param vertices Global vertex indices of entity
/// @return owning process number
template <typename U, typename V>
int get_ownership(const U& processes, const V& vertices)
{
  // Use a deterministic random number generator, seeded with global
  // vertex indices ensuring all processes get the same answer
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
/// @param[in] cell_map Index map for cell distribution
/// @param[in] vertex_map Index map for vertex distribution
/// @param[in] entity_list List of entities, each entity represented by
/// its local vertex indices
/// @param[in] num_vertices_per_e Number of vertices per entity
/// @param[in] num_entities_per_cell Number of entities per cell
/// @param[in] entity_index Initial numbering for each row in
/// entity_list
/// @returns Local indices and index map
std::tuple<std::vector<int>, common::IndexMap>
get_local_indexing(MPI_Comm comm, const common::IndexMap& cell_map,
                   const common::IndexMap& vertex_map,
                   const xtl::span<const std::int32_t>& entity_list,
                   int num_vertices_per_e, int num_entities_per_cell,
                   const xtl::span<const std::int32_t>& entity_index)
{
  // entity_list contains all the entities for all the cells, listed as
  // local vertex indices, and entity_index contains the initial
  // numbering of the entities.
  //                   entity_list entity_index
  // e.g. cell0-ent0: [0,1,2]      15
  //      cell0-ent1: [1,2,3]      23
  //      cell1-ent0: [0,1,2]      15
  //      cell1-ent1: [1,2,6]      24
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
  // 3 = entities with ownership that needs deciding (used also for
  // un-ghosted case)
  std::vector<int> ghost_status(entity_count, 0);
  {
    if (cell_map.num_ghosts() == 0)
      std::fill(ghost_status.begin(), ghost_status.end(), 3);
    else
    {
      const std::int32_t ghost_offset
          = cell_map.size_local() * num_entities_per_cell;

      // Tag all entities in local cells with 1
      for (int i = 0; i < ghost_offset; ++i)
      {
        const std::int32_t idx = entity_index[i];
        ghost_status[idx] = 1;
      }

      // Set entities in ghost cells to 2 (purely ghost) or 3 (border)
      for (std::size_t i = ghost_offset; i < entity_index.size(); ++i)
      {
        const std::int32_t idx = entity_index[i];
        ghost_status[idx] = ghost_status[idx] | 2;
      }
    }
  }

  //---------
  // Create a symmetric neighbor_comm from vertex_ranks

  // Get sharing ranks for each vertex
  graph::AdjacencyList<int> vertex_ranks = vertex_map.index_to_dest_ranks();

  // Create unique list of ranks that share vertices (owners of)
  std::vector<int> ranks(vertex_ranks.array().begin(),
                         vertex_ranks.array().end());
  std::sort(ranks.begin(), ranks.end());
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(
      comm, ranks.size(), ranks.data(), MPI_UNWEIGHTED, ranks.size(),
      ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neighbor_comm);

  std::vector<std::vector<std::int64_t>> send_entities(ranks.size());
  std::vector<std::vector<std::int32_t>> send_index(ranks.size());

  // Get all "possibly shared" entities, based on vertex sharing. Send
  // to other processes, and see if we get the same back.

  // Map from entity (defined by global vertex indices) to local entity
  // index
  std::vector<std::int64_t> entity_to_local_idx;
  std::vector<std::int32_t> perm;

  {
    // If another rank shares all vertices of an entity, it may need the
    // entity Set of sharing procs for each entity, counting vertex hits
    std::vector<std::int64_t> vglobal(num_vertices_per_e);
    std::vector<int> entity_ranks;
    for (auto entity_idx = entity_index.begin();
         entity_idx != entity_index.end(); ++entity_idx)
    {
      // Get entity vertices
      std::size_t pos = std::distance(entity_index.begin(), entity_idx);
      xtl::span entity
          = entity_list.subspan(pos * num_vertices_per_e, num_vertices_per_e);

      // Build list of ranks that share vertices of the entity, and sort
      entity_ranks.clear();
      for (auto v : entity)
      {
        entity_ranks.insert(entity_ranks.end(), vertex_ranks.links(v).begin(),
                            vertex_ranks.links(v).end());
      }
      std::sort(entity_ranks.begin(), entity_ranks.end());

      // If the number of vertices shared with a rank is
      // 'num_vertices_per_e', then add entity data to the send buffer
      auto it = entity_ranks.begin();
      while (it != entity_ranks.end())
      {
        auto it1 = std::find_if(it, entity_ranks.end(),
                                [r0 = *it](auto r1) { return r1 != r0; });
        if (std::distance(it, it1) == num_vertices_per_e)
        {
          vertex_map.local_to_global(entity, vglobal);
          std::sort(vglobal.begin(), vglobal.end());
          entity_to_local_idx.insert(entity_to_local_idx.end(), vglobal.begin(),
                                     vglobal.end());
          entity_to_local_idx.push_back(*entity_idx);

          // Only send entities that are not known to be ghosts
          if (ghost_status[*entity_idx] != 2)
          {
            auto itr_local = std::lower_bound(ranks.begin(), ranks.end(), *it);
            assert(itr_local != ranks.end() and *itr_local == *it);
            const int r = std::distance(ranks.begin(), itr_local);

            // Entity entity_idx may be shared with rank r
            send_entities[r].insert(send_entities[r].end(), vglobal.begin(),
                                    vglobal.end());
            send_index[r].push_back(*entity_idx);
          }
        }

        it = it1;
      }
    }

    perm.resize(entity_to_local_idx.size() / (num_vertices_per_e + 1));
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&entities = entity_to_local_idx,
               shape = num_vertices_per_e + 1](auto e0, auto e1)
              {
                auto it0 = std::next(entities.begin(), e0 * shape);
                auto it1 = std::next(entities.begin(), e1 * shape);
                return std::lexicographical_compare(it0, std::next(it0, shape),
                                                    it1, std::next(it1, shape));
              });
    perm.erase(std::unique(perm.begin(), perm.end(),
                           [&entities = entity_to_local_idx,
                            shape = num_vertices_per_e + 1](auto e0, auto e1)
                           {
                             auto it0 = std::next(entities.begin(), e0 * shape);
                             auto it1 = std::next(entities.begin(), e1 * shape);
                             return std::equal(it0, std::next(it0, shape), it1);
                           }),
               perm.end());
  }

  // Get shared entities of this dimension, and also match up an index
  // for the received entities (from other processes) with the indices
  // of the sent entities (to other processes)

  // Send/receive entities
  std::vector<std::int64_t> recv_data;
  std::vector<int> send_sizes, send_disp, recv_disp, recv_sizes;
  {
    std::vector<std::int64_t> send_buffer;
    for (auto& x : send_entities)
    {
      send_sizes.push_back(x.size());
      send_buffer.insert(send_buffer.end(), x.begin(), x.end());
    }
    assert(send_sizes.size() == ranks.size());

    // Build send displacements
    send_disp = {0};
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::back_inserter(send_disp));

    recv_sizes.resize(ranks.size());
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, neighbor_comm);

    // Build recv displacements
    recv_disp = {0};
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::back_inserter(recv_disp));

    recv_data.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, recv_data.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           neighbor_comm);
  }

  // List of (local index, sorted global vertices) pairs received from
  // othe ranks. The list is eventually sorted.
  std::vector<std::pair<std::int32_t, std::int64_t>>
      shared_entity_to_global_vertices_data;

  // List of (local enity index, global MPI ranks)
  std::vector<std::pair<std::int32_t, int>> shared_entities_data;

  // Compare received and sent entity keys. Any received entities
  // not found in entity_to_local_idx will have recv_index
  // set to -1.
  const int mpi_rank = dolfinx::MPI::rank(comm);
  std::vector<std::int32_t> recv_index;
  recv_index.reserve(recv_disp.size() - 1);
  for (std::size_t r = 0; r < recv_disp.size() - 1; ++r)
  {
    // Loop over received entities (defined by array of entity vertices)
    for (int j = recv_disp[r]; j < recv_disp[r + 1]; j += num_vertices_per_e)
    {
      xtl::span<const std::int64_t> entity(recv_data.data() + j,
                                           num_vertices_per_e);
      auto it = std::lower_bound(
          perm.begin(), perm.end(), entity,
          [&entities = entity_to_local_idx,
           shape = num_vertices_per_e](auto& e0, auto& e1)
          {
            auto it0 = std::next(entities.begin(), e0 * (shape + 1));
            return std::lexicographical_compare(it0, std::next(it0, shape),
                                                e1.begin(), e1.end());
          });

      if (it != perm.end())
      {
        auto offset = (*it) * (num_vertices_per_e + 1);
        xtl::span<const std::int64_t> e(entity_to_local_idx.data() + offset,
                                        num_vertices_per_e + 1);
        if (std::equal(e.begin(), std::prev(e.end()), entity.begin()))
        {
          auto idx = e.back();
          shared_entities_data.push_back({idx, ranks[r]});
          shared_entities_data.push_back({idx, mpi_rank});
          recv_index.push_back(idx);
          std::transform(
              entity.begin(), entity.end(),
              std::back_inserter(shared_entity_to_global_vertices_data),
              [idx](auto v) -> std::pair<std::int32_t, std::int64_t> {
                return {idx, v};
              });
        }
        else
          recv_index.push_back(-1);
      }
      else
        recv_index.push_back(-1);
    }
  }

  const graph::AdjacencyList<int> shared_entities
      = create_adj_list(shared_entities_data, entity_count);

  const graph::AdjacencyList<int> shared_entities_v
      = create_adj_list(shared_entity_to_global_vertices_data, entity_count);

  //---------
  // Determine ownership of shared entities

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
      if (auto ranks = shared_entities.links(i);
          ghost_status[i] == 1 or ranks.empty())
      {
        local_index[i] = c++;
      }
      else
      {
        auto vertices = shared_entities_v.links(i);
        assert(!vertices.empty());
        int owner_rank = get_ownership(ranks, vertices);
        if (owner_rank == mpi_rank)
        {
          // Take ownership
          local_index[i] = c++;
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
    MPI_Exscan(&_num_local, &local_offset, 1, MPI_INT64_T, MPI_SUM, comm);

    // Send global indices for same entities that we sent before. This
    // uses the same pattern as before, so we can match up the received
    // data to the indices in recv_index
    std::vector<std::int64_t> send_global_index_data;
    for (const auto& indices : send_index)
    {
      std::transform(indices.cbegin(), indices.cend(),
                     std::back_inserter(send_global_index_data),
                     [&local_index, size = num_local,
                      offset = local_offset](auto idx) -> std::int64_t
                     {
                       // If not in our local range, send -1.
                       return local_index[idx] < size
                                  ? offset + local_index[idx]
                                  : -1;
                     });
    }

    // Transform send/receive sizes and displacements for scalar send
    for (auto x : {&send_sizes, &send_disp, &recv_sizes, &recv_disp})
    {
      std::transform(x->begin(), x->end(), x->begin(),
                     [num_vertices_per_e](auto a)
                     { return a / num_vertices_per_e; });
    }

    recv_data.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(send_global_index_data.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, recv_data.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           neighbor_comm);
    MPI_Comm_free(&neighbor_comm);

    // Map back received indices
    for (std::size_t r = 0; r < recv_disp.size() - 1; ++r)
    {
      for (int i = recv_disp[r]; i < recv_disp[r + 1]; ++i)
      {
        const std::int64_t gi = recv_data[i];
        const std::int32_t idx = recv_index[i];
        if (gi != -1 and idx != -1)
        {
          assert(local_index[idx] >= num_local);
          ghost_indices[local_index[idx] - num_local] = gi;
          ghost_owners[local_index[idx] - num_local] = ranks[r];
        }
      }
    }

    assert(std::find(ghost_indices.begin(), ghost_indices.end(), -1)
           == ghost_indices.end());
  }

  common::IndexMap index_map(comm, num_local, ghost_indices, ghost_owners);

  // Create map from initial numbering to new local indices
  std::vector<std::int32_t> new_entity_index(entity_index.size());
  std::transform(entity_index.cbegin(), entity_index.cend(),
                 new_entity_index.begin(),
                 [&local_index](auto index) { return local_index[index]; });

  return {std::move(new_entity_index), std::move(index_map)};
}
//-----------------------------------------------------------------------------

/// Compute entities of dimension d
///
/// @param[in] comm MPI communicator (TODO: full or neighbor hood?)
/// @param[in] cells Adjacency list for cell-vertex connectivity
/// @param[in] shared_vertices TODO
/// @param[in] cell_type Cell type
/// @param[in] dim Topological dimension of the entities to be computed
/// @return Returns the (cell-entity connectivity, entity-cell
/// connectivity, index map for the entity distribution across
/// processes, shared entities)
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
  const std::int8_t num_entities_per_cell = cell_num_entities(cell_type, dim);

  // For some cells, the num_vertices varies per facet (3 or 4)
  int max_vertices_per_entity = 0;
  for (int i = 0; i < num_entities_per_cell; ++i)
  {
    max_vertices_per_entity
        = std::max(max_vertices_per_entity,
                   num_cell_vertices(cell_entity_type(cell_type, dim, i)));
  }

  // Create map from cell vertices to entity vertices
  auto e_vertices = get_entity_vertices(cell_type, dim);

  // List of vertices for each entity in each cell
  const std::size_t num_cells = cells.num_nodes();
  const std::size_t entity_list_shape0 = num_cells * num_entities_per_cell;
  const std::size_t entity_list_shape1 = max_vertices_per_entity;
  std::vector<std::int32_t> entity_list(entity_list_shape0 * entity_list_shape1,
                                        -1);
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // Get vertices from cell
    auto vertices = cells.links(c);

    for (int i = 0; i < num_entities_per_cell; ++i)
    {
      const std::int32_t idx = c * num_entities_per_cell + i;
      auto ev = e_vertices.links(i);

      // Get entity vertices. Padded with -1 if fewer than
      // max_vertices_per_entity
      for (std::size_t j = 0; j < ev.size(); ++j)
        entity_list[idx * entity_list_shape1 + j] = vertices[ev[j]];
    }
  }

  std::vector<std::int32_t> entity_index(entity_list_shape0, -1);
  std::int32_t entity_count = 0;
  {
    // Copy list and sort vertices of each entity into (reverse) order
    std::vector<std::int32_t> entity_list_sorted = entity_list;
    for (std::size_t i = 0; i < entity_list_shape0; ++i)
    {
      auto it = std::next(entity_list_sorted.begin(), i * entity_list_shape1);
      std::sort(it, std::next(it, entity_list_shape1), std::greater<>());
    }

    // Sort the list and label uniquely
    const std::vector<std::int32_t> sort_order
        = dolfinx::sort_by_perm<std::int32_t>(entity_list_sorted,
                                              entity_list_shape1);

    std::vector<std::int32_t> entity(max_vertices_per_entity),
        entity0(max_vertices_per_entity);
    auto it = sort_order.begin();
    while (it != sort_order.end())
    {
      // First entity in new index range
      std::size_t offset = (*it) * max_vertices_per_entity;
      xtl::span e0(entity_list_sorted.data() + offset, max_vertices_per_entity);

      // Find iterator to next entity
      auto it1 = std::find_if_not(
          it, sort_order.end(),
          [e0, &entity_list_sorted, max_vertices_per_entity](auto idx) -> bool
          {
            std::size_t offset = idx * max_vertices_per_entity;
            return std::equal(e0.cbegin(), e0.cend(),
                              std::next(entity_list_sorted.begin(), offset));
          });

      // Set entity unique index
      std::for_each(it, it1,
                    [&entity_index, entity_count](auto idx)
                    { entity_index[idx] = entity_count; });

      // Advance iterator and increment entity
      it = it1;
      ++entity_count;
    }
  }

  // Communicate with other processes to find out which entities are
  // ghosted and shared. Remap the numbering so that ghosts are at the
  // end.
  auto [local_index, index_map] = get_local_indexing(
      comm, cell_index_map, vertex_index_map, entity_list,
      max_vertices_per_entity, num_entities_per_cell, entity_index);

  // Entity-vertex connectivity
  std::vector<std::int32_t> offsets_ev(entity_count + 1, 0);
  std::vector<int> size_ev(entity_count);
  for (std::size_t i = 0; i < entity_list_shape0; ++i)
  {
    // if (entity_list(i, max_vertices_per_entity - 1) == -1)
    if (entity_list[i * entity_list_shape1 + entity_list_shape1 - 1] == -1)
      size_ev[local_index[i]] = max_vertices_per_entity - 1;
    else
      size_ev[local_index[i]] = max_vertices_per_entity;
  }

  std::transform(size_ev.cbegin(), size_ev.cend(), offsets_ev.cbegin(),
                 std::next(offsets_ev.begin()),
                 [](auto a, auto b) { return a + b; });

  graph::AdjacencyList<std::int32_t> ev(
      std::vector<std::int32_t>(offsets_ev.back()), std::move(offsets_ev));
  for (std::size_t i = 0; i < entity_list_shape0; ++i)
  {
    auto _ev = ev.links(local_index[i]);
    std::copy_n(std::next(entity_list.begin(), i * entity_list_shape1),
                _ev.size(), _ev.begin());
  }

  // NOTE: Cell-entity connectivity comes after ev creation because
  // below we use std::move(local_index)

  // Cell-entity connectivity
  std::vector<std::int32_t> offsets_ce(num_cells + 1, 0);
  std::transform(offsets_ce.cbegin(), std::prev(offsets_ce.cend()),
                 std::next(offsets_ce.begin()),
                 [num_entities_per_cell](auto x)
                 { return x + num_entities_per_cell; });
  graph::AdjacencyList<std::int32_t> ce(std::move(local_index),
                                        std::move(offsets_ce));

  return {std::move(ce), std::move(ev), std::move(index_map)};
}
//-----------------------------------------------------------------------------

/// Compute connectivity from entities of dimension d0 to entities of
/// dimension d1 using the transpose connectivity (d1 -> d0)
///
/// @param[in] c_d1_d0 The connectivity from entities of dimension d1 to
/// entities of dimension d0.
/// @param[in] num_entities_d0 The number of entities of dimension d0.
/// @return The connectivity from entities of dimension d0 to entities
/// of dimension d1.
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

  // Number of edges for a tri/quad is the same as number of vertices so
  // AdjacencyList will have same offset pattern
  std::vector<std::int32_t> connections;
  connections.reserve(c_d0_0.array().size());
  std::vector<std::int32_t> offsets(c_d0_0.offsets());

  // Search for edges of facet in map, and recover index
  const auto tri_vertices_ref
      = get_entity_vertices(mesh::CellType::triangle, 1);
  const auto quad_vertices_ref
      = get_entity_vertices(mesh::CellType::quadrilateral, 1);

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
