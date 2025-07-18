// Copyright (C) 2006-2024 Anders Logg, Garth N. Wells and Chris Richardson
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
#include <mpi.h>
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
  std::ranges::sort(data);
  auto [unique_end, range_end] = std::ranges::unique(data);
  data.erase(unique_end, range_end);

  std::vector<int> array;
  array.reserve(data.size());
  std::ranges::transform(data, std::back_inserter(array),
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

  return graph::AdjacencyList(std::move(array), std::move(offsets));
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
/// @returns Local indices, the index map and shared entities
std::tuple<std::vector<int>, common::IndexMap, std::vector<std::int32_t>>
get_local_indexing(MPI_Comm comm, const common::IndexMap& vertex_map,
                   std::span<const std::int32_t> entity_list,
                   int num_vertices_per_e,
                   const std::vector<std::int8_t>& ghost_status,
                   std::span<const std::int32_t> entity_index)
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
  if (auto mx = std::ranges::max_element(entity_index);
      mx != entity_index.end())
  {
    entity_count = *mx + 1;
  }

  //---------
  // Create a symmetric neighbor_comm from vertex_ranks

  // Get sharing ranks for each vertex
  graph::AdjacencyList<int> vertex_ranks = vertex_map.index_to_dest_ranks();

  // Create unique list of ranks that share vertices (owners of)
  std::vector<int> ranks(vertex_ranks.array().begin(),
                         vertex_ranks.array().end());
  std::ranges::sort(ranks);
  auto [unique_end, range_end] = std::ranges::unique(ranks);
  ranks.erase(unique_end, range_end);

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
      std::span entity
          = entity_list.subspan(pos * num_vertices_per_e, num_vertices_per_e);

      // Build list of ranks that share vertices of the entity, and sort
      entity_ranks.clear();
      for (auto v : entity)
      {
        entity_ranks.insert(entity_ranks.end(), vertex_ranks.links(v).begin(),
                            vertex_ranks.links(v).end());
      }
      std::ranges::sort(entity_ranks);

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
          std::ranges::sort(vglobal);
          entity_to_local_idx.insert(entity_to_local_idx.end(), vglobal.begin(),
                                     vglobal.end());
          entity_to_local_idx.push_back(*entity_idx);

          // Only send entities that are not known to be ghosts
          if (ghost_status[*entity_idx] != 1)
          {
            auto itr_local = std::ranges::lower_bound(ranks, *it);
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

    auto range_by_index = [&, shape = num_vertices_per_e + 1](auto e)
    {
      auto begin = std::next(entity_to_local_idx.begin(), e * shape);
      return std::ranges::subrange(begin, std::next(begin, shape));
    };

    std::ranges::sort(perm, std::ranges::lexicographical_compare,
                      range_by_index);

    auto [unique_end, range_end]
        = std::ranges::unique(perm, std::ranges::equal, range_by_index);

    perm.erase(unique_end, range_end);
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
  // other ranks. The list is eventually sorted.
  std::vector<std::pair<std::int32_t, std::int64_t>>
      shared_entity_to_global_vertices_data;

  // List of (local entity index, global MPI ranks)
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
      std::span<const std::int64_t> entity(recv_data.data() + j,
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
        std::span<const std::int64_t> e(entity_to_local_idx.data() + offset,
                                        num_vertices_per_e + 1);
        if (std::equal(e.begin(), std::prev(e.end()), entity.begin()))
        {
          auto idx = e.back();
          shared_entities_data.push_back({idx, ranks[r]});
          shared_entities_data.push_back({idx, mpi_rank});
          recv_index.push_back(idx);
          std::ranges::transform(
              entity, std::back_inserter(shared_entity_to_global_vertices_data),
              [idx](auto v) -> std::pair<std::int32_t, std::int64_t>
              { return {idx, v}; });
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
  std::vector<std::int32_t> interprocess_entities;
  std::int32_t num_local;
  {
    std::int32_t c = 0;

    // Index non-ghost entities
    for (int i = 0; i < entity_count; ++i)
    {
      // Definitely ghost
      if (ghost_status[i] == 1)
        continue;

      if (auto ranks = shared_entities.links(i); ranks.empty())
      {
        // Definitely local, unshared
        local_index[i] = c++;
      }
      else
      {
        // Shared with another process
        interprocess_entities.push_back(i);
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

    std::ranges::transform(local_index, local_index.begin(), [&c](auto index)
                           { return index == -1 ? c++ : index; });
    assert(c == entity_count);

    // Convert interprocess entities to local_index
    std::ranges::transform(interprocess_entities, interprocess_entities.begin(),
                           [&local_index](auto i) { return local_index[i]; });
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
      std::ranges::transform(
          indices, std::back_inserter(send_global_index_data),
          [&local_index, size = num_local,
           offset = local_offset](auto idx) -> std::int64_t
          {
            // If not in our local range, send -1.
            return local_index[idx] < size ? offset + local_index[idx] : -1;
          });
    }

    // Transform send/receive sizes and displacements for scalar send
    for (auto x : {&send_sizes, &send_disp, &recv_sizes, &recv_disp})
      std::ranges::transform(*x, x->begin(), [num_vertices_per_e](auto a)
                             { return a / num_vertices_per_e; });

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
  std::ranges::transform(entity_index, new_entity_index.begin(),
                         [&local_index](auto index)
                         { return local_index[index]; });

  return {std::move(new_entity_index), std::move(index_map),
          std::move(interprocess_entities)};
}
//-----------------------------------------------------------------------------

/// Compute entities of dimension d
///
/// @param[in] comm MPI communicator (TODO: full or neighbor hood?)
/// @param[in] cells Adjacency list for cell-vertex connectivity
/// @param[in] shared_vertices TODO
/// @param[in] cell_type Cell type
/// @param[in] dim Topological dimension of the entities to be computed
/// @return Returns the (cell-entity connectivity, entity-vertex
/// connectivity, index map for the entity distribution across
/// processes, shared entities)
std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
           graph::AdjacencyList<std::int32_t>, common::IndexMap,
           std::vector<std::int32_t>>
compute_entities_by_key_matching(
    MPI_Comm comm,
    std::vector<
        std::tuple<mesh::CellType,
                   std::shared_ptr<const graph::AdjacencyList<std::int32_t>>,
                   std::shared_ptr<const common::IndexMap>>>
        cell_lists,
    const common::IndexMap& vertex_index_map, mesh::CellType entity_type,
    int dim)
{
  if (dim == 0)
  {
    throw std::runtime_error(
        "Cannot create vertices for topology. Should already exist.");
  }

  assert(cell_dim(entity_type) == dim);

  // Start timer
  common::Timer timer("Compute entities of dim = " + std::to_string(dim));

  std::vector<std::vector<std::int32_t>> cell_type_entities(cell_lists.size());
  std::vector<std::int32_t> cell_type_offsets{0};
  for (std::size_t k = 0; k < cell_lists.size(); ++k)
  {
    mesh::CellType cell_type = std::get<0>(cell_lists[k]);
    auto cells = std::get<1>(cell_lists[k]);
    const std::size_t num_cells = cells->num_nodes();

    for (int i = 0; i < cell_num_entities(cell_type, dim); ++i)
    {
      if (cell_entity_type(cell_type, dim, i) == entity_type)
        cell_type_entities[k].push_back(i);
    }
    cell_type_offsets.push_back(cell_type_offsets.back()
                                + num_cells * cell_type_entities[k].size());
  }

  int num_vertices_per_entity = num_cell_vertices(entity_type);
  std::vector<std::int32_t> entity_list(cell_type_offsets.back()
                                        * num_vertices_per_entity);

  for (std::size_t k = 0; k < cell_lists.size(); ++k)
  {
    auto cell_type = std::get<0>(cell_lists[k]);
    auto cells = std::get<1>(cell_lists[k]);
    auto cell_index_map = std::get<2>(cell_lists[k]);

    // Get indices of desired entities within cell. Usually this will be all
    // entities, but for prism or pyramid facets, we will just pick out
    // triangle or quad facets.

    // Create map from cell vertices to entity vertices
    auto e_vertices = get_entity_vertices(cell_type, dim);

    const std::size_t num_cells = cells->num_nodes();
    int num_entities_per_cell = cell_type_entities[k].size();
    for (std::size_t c = 0; c < num_cells; ++c)
    {
      // Get vertices from each cell
      auto vertices = cells->links(c);

      for (int i = 0; i < num_entities_per_cell; ++i)
      {
        const std::int32_t idx = c * num_entities_per_cell + i;
        auto ev = e_vertices.links(cell_type_entities[k][i]);

        // Get entity vertices. Padded with -1 if fewer than
        // max_vertices_per_entity
        // NOTE Entity orientation is determined by vertex ordering. The
        // orientation of an entity with respect to the cell may differ from its
        // global mesh orientation. Hence, we reorder the vertices so that
        // each entity's orientation agrees with their global orientation.
        // FIXME This might be better below when the entity to vertex
        // connectivity is computed
        std::vector<std::int32_t> entity_vertices(ev.size());
        for (std::size_t j = 0; j < ev.size(); ++j)
          entity_vertices[j] = vertices[ev[j]];

        // Orient the entities. Simply sort according to global vertex index
        // for simplices
        std::vector<std::int64_t> global_vertices(entity_vertices.size());
        vertex_index_map.local_to_global(entity_vertices, global_vertices);

        std::vector<std::size_t> perm(global_vertices.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::ranges::sort(
            perm, [&global_vertices](std::size_t i0, std::size_t i1)
            { return global_vertices[i0] < global_vertices[i1]; });
        // For quadrilaterals, the vertex opposite the lowest vertex should
        // be last
        if (entity_type == mesh::CellType::quadrilateral)
        {
          std::size_t min_vertex_idx = perm[0];
          std::size_t opposite_vertex_index = 3 - min_vertex_idx;
          auto it = std::find(perm.begin(), perm.end(), opposite_vertex_index);
          assert(it != perm.end());
          std::rotate(it, it + 1, perm.end());
        }

        for (std::size_t j = 0; j < ev.size(); ++j)
          entity_list[(cell_type_offsets[k] + idx) * num_vertices_per_entity
                      + j]
              = entity_vertices[perm[j]];
      }
    }
  }

  // Start numbering entities
  std::vector<std::int32_t> entity_index(cell_type_offsets.back());

  std::int32_t entity_count = 0;
  {
    // Copy list and sort vertices of each entity into (reverse) order
    std::vector<std::int32_t> entity_list_sorted = entity_list;
    for (std::size_t j = 0; j < entity_index.size(); ++j)
    {
      auto it
          = std::next(entity_list_sorted.begin(), j * num_vertices_per_entity);
      std::sort(it, std::next(it, num_vertices_per_entity), std::less<>());
    }

    // Sort the list and label uniquely
    const std::vector<std::int32_t> sort_order
        = dolfinx::sort_by_perm<std::int32_t>(entity_list_sorted,
                                              num_vertices_per_entity);

    std::vector<std::int32_t> entity(num_vertices_per_entity),
        entity0(num_vertices_per_entity);
    auto it = sort_order.begin();
    while (it != sort_order.end())
    {
      // First entity in new index range
      std::size_t offset = (*it) * num_vertices_per_entity;
      std::span e0(entity_list_sorted.data() + offset, num_vertices_per_entity);

      // Find iterator to next entity
      auto it1 = std::find_if_not(
          it, sort_order.end(),
          [e0, &entity_list_sorted, num_vertices_per_entity](auto idx) -> bool
          {
            std::size_t offset = idx * num_vertices_per_entity;
            return std::equal(e0.begin(), e0.end(),
                              std::next(entity_list_sorted.begin(), offset));
          });

      // Set entity unique index
      std::for_each(it, it1, [&entity_index, entity_count](auto idx)
                    { entity_index[idx] = entity_count; });

      // Advance iterator and increment entity
      it = it1;
      ++entity_count;
    }
  }

  //---------
  // Set ghost status array values
  // 0 = entities that are only in ghost cells (i.e. definitely not owned)
  // 1 = entities with local ownership or ownership that needs deciding
  std::vector<std::int8_t> ghost_status(entity_count, 1);
  for (std::size_t k = 0; k < cell_lists.size(); ++k)
  {
    auto cells = std::get<1>(cell_lists[k]);
    [[maybe_unused]] const std::size_t num_cells = cells->num_nodes();
    auto cell_map = std::get<2>(cell_lists[k]);
    int num_entities_per_cell = cell_type_entities[k].size();
    assert(cell_map->size_local() + cell_map->num_ghosts() == (int)num_cells);

    const std::int32_t ghost_offset = cell_map->size_local();
    // Tag all entities in local cells with 0, leaving entities which only
    // appear in ghost cells tagged.
    for (std::int32_t i = 0; i < ghost_offset * num_entities_per_cell; ++i)
    {
      const std::int32_t idx = entity_index[i + cell_type_offsets[k]];
      ghost_status[idx] = 0;
    }
  }

  // Communicate with other processes to find out which entities are
  // ghosted and shared. Remap the numbering so that ghosts are at the
  // end.

  auto [local_index, index_map, interprocess_entities]
      = get_local_indexing(comm, vertex_index_map, entity_list,
                           num_vertices_per_entity, ghost_status, entity_index);

  // Entity-vertex connectivity
  std::vector<std::int32_t> ev_array(entity_count * num_vertices_per_entity);
  graph::AdjacencyList ev = graph::regular_adjacency_list(
      std::move(ev_array), num_vertices_per_entity);
  for (std::int32_t i = 0; i < cell_type_offsets.back(); ++i)
  {
    std::copy_n(std::next(entity_list.begin(), i * num_vertices_per_entity),
                num_vertices_per_entity, ev.links(local_index[i]).begin());
  }

  std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>> ce(
      cell_lists.size());
  for (std::size_t k = 0; k < cell_lists.size(); ++k)
  {

    if (!cell_type_entities[k].empty())
    {
      std::vector tmp(std::next(local_index.begin(), cell_type_offsets[k]),
                      std::next(local_index.begin(), cell_type_offsets[k + 1]));
      ce[k] = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          graph::regular_adjacency_list(std::move(tmp),
                                        cell_type_entities[k].size()));
    }
  }

  return {ce, std::move(ev), std::move(index_map),
          std::move(interprocess_entities)};
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
                       const int num_entities_d0)
{

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

  return graph::AdjacencyList(std::move(connections), std::move(offsets));
}
//-----------------------------------------------------------------------------

/// Compute the d0 -> d1 connectivity, where d0 > d1
/// @param[in] c_d0_0 The d0 -> 0 (entity (d0) to vertex) connectivity
/// @param[in] c_d0_0 The d1 -> 0 (entity (d1) to vertex) connectivity
/// @param[in] cell_type_d0 The cell type for entities of dimension d0
/// @return The d0 -> d1 connectivity
graph::AdjacencyList<std::int32_t>
compute_from_map(const graph::AdjacencyList<std::int32_t>& c_d0_0,
                 const graph::AdjacencyList<std::int32_t>& c_d1_0)
{
  // Make a map from the sorted edge vertices to the edge index
  boost::unordered_map<std::array<std::int32_t, 2>, std::int32_t> edge_to_index;
  edge_to_index.reserve(c_d1_0.num_nodes());

  std::array<std::int32_t, 2> key;
  for (int e = 0; e < c_d1_0.num_nodes(); ++e)
  {
    std::span<const std::int32_t> v = c_d1_0.links(e);
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
  const graph::AdjacencyList<int> tri_vertices_ref
      = get_entity_vertices(mesh::CellType::triangle, 1);
  const graph::AdjacencyList<int> quad_vertices_ref
      = get_entity_vertices(mesh::CellType::quadrilateral, 1);
  for (int e = 0; e < c_d0_0.num_nodes(); ++e)
  {
    auto e0 = c_d0_0.links(e);
    auto vref = (e0.size() == 3) ? &tri_vertices_ref : &quad_vertices_ref;
    for (std::size_t i = 0; i < e0.size(); ++i)
    {
      auto v = vref->links(i);
      for (int j = 0; j < 2; ++j)
        key[j] = e0[v[j]];
      std::ranges::sort(key);
      auto it = edge_to_index.find(key);
      assert(it != edge_to_index.end());
      connections.push_back(it->second);
    }
  }

  connections.shrink_to_fit();
  return graph::AdjacencyList(std::move(connections), std::move(offsets));
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>, std::vector<std::int32_t>>
mesh::compute_entities(const Topology& topology, int dim, CellType entity_type)
{
  spdlog::info("Computing mesh entities of dimension {}", dim);

  // Vertices must always exist
  if (dim == 0)
  {
    return {std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>(),
            nullptr, nullptr, std::vector<std::int32_t>()};
  }

  {
    auto idx = std::ranges::find(topology.entity_types(dim), entity_type);
    assert(idx != topology.entity_types(dim).end());
    int index = std::distance(topology.entity_types(dim).begin(), idx);
    if (topology.connectivity({dim, index}, {0, 0}))
    {
      return {
          std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>(),
          nullptr, nullptr, std::vector<std::int32_t>()};
    }
  }

  const int tdim = topology.dim();

  // Lists of all cells by cell type
  std::vector<CellType> cell_types = topology.entity_types(tdim);
  std::vector<std::tuple<
      mesh::CellType, std::shared_ptr<const graph::AdjacencyList<std::int32_t>>,
      std::shared_ptr<const common::IndexMap>>>
      cell_lists(cell_types.size());

  auto cell_index_maps = topology.index_maps(tdim);
  for (std::size_t i = 0; i < cell_types.size(); ++i)
  {
    auto cell_map = cell_index_maps[i];
    assert(cell_map);
    auto cells = topology.connectivity({tdim, int(i)}, {0, 0});
    if (!cells)
      throw std::runtime_error("Cell connectivity missing.");
    cell_lists[i] = {cell_types[i], cells, cell_map};
  }

  auto vertex_map = topology.index_map(0);
  assert(vertex_map);

  // c->e, e->v
  auto [d0, d1, im, interprocess_entities] = compute_entities_by_key_matching(
      topology.comm(), cell_lists, *vertex_map, entity_type, dim);

  return {d0,
          std::make_shared<graph::AdjacencyList<std::int32_t>>(std::move(d1)),
          std::make_shared<common::IndexMap>(std::move(im)),
          std::move(interprocess_entities)};
}
//-----------------------------------------------------------------------------
std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
mesh::compute_connectivity(const Topology& topology, std::array<int, 2> d0,
                           std::array<int, 2> d1)
{
  spdlog::info("Requesting connectivity ({}, {}) - ({}, {})",
               std::to_string(d0[0]), std::to_string(d0[1]),
               std::to_string(d1[0]), std::to_string(d1[1]));

  // Return if connectivity has already been computed
  if (topology.connectivity(d0, d1))
    return {nullptr, nullptr};

  // Return if no connectivity is possible
  if (d0[0] == d1[0] and d0[1] != d1[1])
    return {nullptr, nullptr};

  // No connectivity between these cell types
  CellType c0 = topology.entity_types(d0[0])[d0[1]];
  CellType c1 = topology.entity_types(d1[0])[d1[1]];
  if ((c0 == CellType::hexahedron and c1 == CellType::triangle)
      or (c0 == CellType::triangle and c1 == CellType::hexahedron))
  {
    return {nullptr, nullptr};
  }
  if ((c0 == CellType::tetrahedron and c1 == CellType::quadrilateral)
      or (c0 == CellType::quadrilateral and c1 == CellType::tetrahedron))
  {
    return {nullptr, nullptr};
  }

  // Get entities if they exist
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d0_0
      = topology.connectivity(d0, {0, 0});
  if (d0[0] > 0 and !topology.connectivity(d0, {0, 0}))
  {
    throw std::runtime_error("Missing entities of dimension "
                             + std::to_string(d0[0]) + ".");
  }

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d1_0
      = topology.connectivity(d1, {0, 0});
  if (d1[0] > 0 and !topology.connectivity(d1, {0, 0}))
  {
    throw std::runtime_error("Missing entities of dimension "
                             + std::to_string(d1[0]) + ".");
  }

  // Start timer
  common::Timer timer("Compute connectivity " + std::to_string(d0[0]) + "-"
                      + std::to_string(d1[1]));

  // Decide how to compute the connectivity
  if (d0 == d1)
  {
    return {std::make_shared<graph::AdjacencyList<std::int32_t>>(
                c_d0_0->num_nodes()),
            nullptr};
  }
  else if (d0[0] < d1[0])
  {
    // Compute connectivity d1 - d0 (if needed), and take transpose
    if (!topology.connectivity(d1, d0))
    {
      // Only possible case is edge->facet
      assert(d0[0] == 1 and d1[0] == 2);
      auto c_d1_d0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_map(*c_d1_0, *c_d0_0));

      spdlog::info("Computing mesh connectivity {}-{} from transpose.", d0[0],
                   d1[0]);
      auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_transpose(*c_d1_d0, c_d0_0->num_nodes()));
      return {c_d0_d1, c_d1_d0};
    }
    else
    {
      assert(c_d0_0);
      assert(topology.connectivity(d1, d0));

      spdlog::info("Computing mesh connectivity {}-{} from transpose.",
                   std::to_string(d0[0]), std::to_string(d1[0]));
      auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_transpose(*topology.connectivity(d1, d0),
                                 c_d0_0->num_nodes()));
      return {c_d0_d1, nullptr};
    }
  }
  else if (d0[0] > d1[0])
  {
    // Compute by mapping vertices from a lower dimension entity to
    // those of a higher dimension entity

    // Only possible case is facet->edge
    assert(d0[0] == 2 and d1[0] == 1);
    auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        compute_from_map(*c_d0_0, *c_d1_0));
    return {c_d0_d1, nullptr};
  }
  else
    throw std::runtime_error("Entity dimension error when computing topology.");
}
//--------------------------------------------------------------------------
