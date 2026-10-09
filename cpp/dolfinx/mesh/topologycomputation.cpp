// Copyright (C) 2006-2026 Anders Logg, Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "topologycomputation.h"
#include "Topology.h"
#include "cell_types.h"
#include <algorithm>
#include <array>
#include <boost/sort/sort.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cassert>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{
/// @brief  Build list of entities (defined by vertices) of a given
/// type from cells.
///
/// Builds `entity_list=[e0_v0, e0_v1, ..., e1_v0, e1_v1, ...]`, by
/// iterating over each cell and for each cell iterating over each
/// entity of type `entity_type`.
///
/// This code is thread-safe.
///
/// @param[in,out] entity_list Flattened output array to write entity
/// vertices into.
/// @param[in] entity_offset Global index of the first entity this call
/// writes into `entity_list_sorted`.
/// @param[in,out] entity_list_sorted Sorted-key columns (column-major,
/// one span per vertex), written at rows `entity_offset` onwards.
/// @param[in] cells Cell-to-vertex connectivity, flattened.
/// @param[in] num_cell_vertices Number of vertices per cell.
/// @param[in] e_vertices Entity-to-vertices, where
/// `e_vertices.links(e)[i]` is the `i`th local (to the cell) vertex
/// index for entity `e`.
/// @param[in] entity_type Type of entity to extract.
/// @param[in] cell_type_entities Indices of entities of type `entity_type`
/// @param[in] vertex_index_map Index map for the vertices.
auto build_entity_list
    = [](std::span<std::int32_t> entity_list, std::size_t entity_offset,
         std::span<const std::span<std::int32_t>> entity_list_sorted,
         std::span<const std::int32_t> cells, std::size_t num_cell_vertices,
         const graph::AdjacencyList<std::int32_t>& e_vertices,
         mesh::CellType entity_type,
         const std::vector<std::int32_t>& cell_type_entities,
         const common::IndexMap& vertex_index_map)
{
  int num_vertices_per_entity = mesh::num_cell_vertices(entity_type);
  int num_entities_per_cell = cell_type_entities.size();

  std::vector<std::int32_t> entity_vertices(num_vertices_per_entity);
  std::vector<std::int64_t> global_vertices(num_vertices_per_entity);
  std::vector<std::size_t> perm(num_vertices_per_entity);

  // Scratch row for the sorted key -- computed exactly as before, then
  // scattered out to entity_list_sorted's column-major storage once
  // complete. 4 covers the largest supported entity (quadrilateral).
  std::array<std::int32_t, 4> row_sorted_storage;

  // Iterate over cells
  auto it_e = entity_list.begin();
  std::size_t entity_idx = entity_offset;
  std::size_t num_cells = cells.size() / num_cell_vertices;
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // Get vertices for cell
    auto vertices = cells.subspan(c * num_cell_vertices, num_cell_vertices);

    // Iterate over cell entities of given type
    for (int e = 0; e < num_entities_per_cell; ++e)
    {
      auto ev = e_vertices.links(cell_type_entities[e]);

      // Get entity vertices. Padded with -1 if fewer than
      // max_vertices_per_entity
      //
      // NOTE: Entity orientation is determined by vertex
      // ordering. The orientation of an entity with respect to
      // the cell may differ from its global mesh orientation.
      // Hence, we reorder the vertices so that each entity's
      // orientation agrees with their global orientation.
      //
      // FIXME: This might be better below when the entity to
      // vertex connectivity is computed
      assert(ev.size() == entity_vertices.size());
      for (std::size_t j = 0; j < ev.size(); ++j)
        entity_vertices[j] = vertices[ev[j]];

      // Orient the entities. Simply sort according to global
      // vertex index for simplices.
      assert(entity_vertices.size() == global_vertices.size());
      vertex_index_map.local_to_global(entity_vertices, global_vertices);

      auto elist = std::span(it_e, num_vertices_per_entity);
      auto elist_sorted
          = std::span(row_sorted_storage.data(), num_vertices_per_entity);

      // Edges and triangles (by far the most common entity types, and
      // never subject to the quadrilateral re-orientation rule below)
      // are hand-unrolled: a generic std::ranges::sort of a 2- or
      // 3-element array is disproportionately expensive when called
      // once per entity instance over a very large mesh.
      if (num_vertices_per_entity == 2)
      {
        std::int32_t l0 = entity_vertices[0], l1 = entity_vertices[1];
        if (global_vertices[0] <= global_vertices[1])
        {
          elist[0] = l0;
          elist[1] = l1;
        }
        else
        {
          elist[0] = l1;
          elist[1] = l0;
        }
        elist_sorted[0] = std::min(elist[0], elist[1]);
        elist_sorted[1] = std::max(elist[0], elist[1]);
      }
      else if (num_vertices_per_entity == 3)
      {
        std::int32_t l0 = entity_vertices[0], l1 = entity_vertices[1],
                     l2 = entity_vertices[2];
        std::int64_t g0 = global_vertices[0], g1 = global_vertices[1],
                     g2 = global_vertices[2];
        if (g0 > g1)
        {
          std::swap(g0, g1);
          std::swap(l0, l1);
        }
        if (g1 > g2)
        {
          std::swap(g1, g2);
          std::swap(l1, l2);
        }
        if (g0 > g1)
        {
          std::swap(g0, g1);
          std::swap(l0, l1);
        }
        elist[0] = l0;
        elist[1] = l1;
        elist[2] = l2;

        std::int32_t s0 = l0, s1 = l1, s2 = l2;
        if (s0 > s1)
          std::swap(s0, s1);
        if (s1 > s2)
          std::swap(s1, s2);
        if (s0 > s1)
          std::swap(s0, s1);
        elist_sorted[0] = s0;
        elist_sorted[1] = s1;
        elist_sorted[2] = s2;
      }
      else
      {
        std::iota(perm.begin(), perm.end(), 0);
        if (perm.size() == 4)
        {
          // Quadrilaterals: a 5-compare-exchange sorting network,
          // equivalent to std::ranges::sort below for the always-distinct
          // keys here (proven equivalent by exhaustive random-trial
          // testing), avoiding the overhead of the general-purpose
          // algorithm in this hot loop. Only the sort step itself is
          // replaced; the quadrilateral re-orientation logic below is
          // unchanged.
          auto cmpswap = [&perm, &global_vertices](std::size_t a, std::size_t b)
          {
            if (global_vertices[perm[a]] > global_vertices[perm[b]])
              std::swap(perm[a], perm[b]);
          };
          cmpswap(0, 1);
          cmpswap(2, 3);
          cmpswap(0, 2);
          cmpswap(1, 3);
          cmpswap(1, 2);
        }
        else
        {
          std::ranges::sort(
              perm, [&global_vertices](auto i0, auto i1)
              { return global_vertices[i0] < global_vertices[i1]; });
        }

        // For quadrilaterals, the vertex opposite the lowest
        // vertex should be last
        if (entity_type == mesh::CellType::quadrilateral)
        {
          std::size_t min_vertex_idx = perm[0];
          std::size_t opposite_vertex_index = 3 - min_vertex_idx;
          auto it = std::find(perm.begin(), perm.end(), opposite_vertex_index);
          assert(it != perm.end());
          std::rotate(it, it + 1, perm.end());
        }

        for (std::size_t j = 0; j < ev.size(); ++j)
          elist[j] = entity_vertices[perm[j]];

        std::ranges::copy(elist, elist_sorted.begin());

        // No supported CellType has an entity with more than 4
        // vertices, so `row_sorted_storage` (fixed-size, avoiding a
        // heap allocation per entity) is sized accordingly -- guard
        // against a future entity type silently overflowing it,
        // rather than relying on the sorting network below to be
        // reached only for size 4.
        if (elist_sorted.size() != row_sorted_storage.size())
          throw std::runtime_error("Unsupported entity vertex count.");

        auto cmpswap_val = [&elist_sorted](std::size_t a, std::size_t b)
        {
          if (elist_sorted[a] > elist_sorted[b])
            std::swap(elist_sorted[a], elist_sorted[b]);
        };
        cmpswap_val(0, 1);
        cmpswap_val(2, 3);
        cmpswap_val(0, 2);
        cmpswap_val(1, 3);
        cmpswap_val(1, 2);
      }

      // Scatter the sorted key row out to its column-major positions.
      for (int k = 0; k < num_vertices_per_entity; ++k)
        entity_list_sorted[k][entity_idx] = elist_sorted[k];

      std::advance(it_e, num_vertices_per_entity);
      ++entity_idx;
    }
  }
};

/// @brief Create an adjacency list from array of pairs, where the first
/// value in the pair is the node and the second value is the edge.
///
/// @param[in] data List of pairs.
/// @param[in] size Number of edges in the graph. For example, this can
/// be used to build an adjacency list that includes 'owned' nodes only.
/// @pre The `data` array must be sorted.
template <typename U>
graph::AdjacencyList<int> create_adj_list(U& data, std::int32_t size)
{
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
    offsets.push_back(offsets.back() + std::ranges::distance(it, it1));
    it = it1;
  }

  return graph::AdjacencyList(std::move(array), std::move(offsets));
}

//-----------------------------------------------------------------------------

/// @brief Get the ownership of an entity shared over several processes.
///
/// @param processes Set of sharing processes.
/// @param vertices Global vertex indices of entity.
/// @return Owning rank (process) index.
template <typename U, typename V>
int get_ownership(const U& processes, const V& vertices)
{
  // Deterministic selection from the global vertex indices, ensuring
  // all processes get the same answer. A plain FNV-1a hash of the
  // vertices is used instead of a seeded std::mt19937: this function
  // is called once per shared entity (so up to millions of times for
  // a large, highly-ghosted mesh), and re-seeding a std::mt19937 (~2.5
  // kB of internal state) via std::seed_seq on every call - needed
  // only to extract a single index - was the dominant cost of entity
  // ownership determination at scale.
  std::uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a offset basis
  for (auto v : vertices)
  {
    h ^= static_cast<std::uint64_t>(v);
    h *= 0x100000001b3ULL; // FNV-1a prime
  }
  // Index directly into the (already contiguous/sized) input range,
  // rather than copying it into a fresh vector -- this function is
  // called once per shared entity, so up to millions of times for a
  // large, highly-ghosted mesh.
  int index = static_cast<int>(h % processes.size());
  int owner = processes[index];
  return owner;
}
//-----------------------------------------------------------------------------

/// @brief Find, for the entities `[e0, e1)`, the neighbourhood ranks
/// that share all of an entity's vertices and so may hold the entity
/// too.
///
/// This code is thread-safe: calls write only their own output
/// buffers. Entities are visited in increasing index, so appending
/// the buffers of consecutive ranges gives what one serial call over
/// all entities gives.
///
/// @param[in] e0 First entity to consider.
/// @param[in] e1 One past the last entity to consider.
/// @param[in] first_instance An instance of each entity.
/// @param[in] entity_list Vertices of each instance, flattened.
/// @param[in] num_vertices_per_e Number of vertices per entity.
/// @param[in] vertex_ranks Neighbourhood ranks sharing each vertex.
/// @param[in] vertex_map Index map for the vertex distribution.
/// @param[in] ghost_status Ghost status of each entity.
/// @param[out] entity_to_local_idx Rows of `[global vertices..., entity
/// index]`, one per (entity, sharing rank) candidate.
/// @param[out] send_entities Global vertices of the candidates to send
/// to each neighbourhood rank.
/// @param[out] send_index Entity index of each entry of
/// `send_entities`.
void build_candidates(std::int32_t e0, std::int32_t e1,
                      std::span<const std::int32_t> first_instance,
                      std::span<const std::int32_t> entity_list,
                      int num_vertices_per_e,
                      const graph::AdjacencyList<int>& vertex_ranks,
                      const common::IndexMap& vertex_map,
                      std::span<const std::int8_t> ghost_status,
                      std::vector<std::int64_t>& entity_to_local_idx,
                      std::vector<std::vector<std::int64_t>>& send_entities,
                      std::vector<std::vector<std::int32_t>>& send_index)
{
  std::vector<std::int64_t> vglobal(num_vertices_per_e);
  std::vector<int> entity_ranks;
  for (std::int32_t id = e0; id < e1; ++id)
  {
    // Get entity vertices (any instance of this entity has the same,
    // globally-oriented, vertex list)
    std::size_t pos = first_instance[id];
    std::span entity
        = entity_list.subspan(pos * num_vertices_per_e, num_vertices_per_e);

    // Build list of neighbourhood ranks that share vertices of the
    // entity, and sort
    entity_ranks.clear();
    for (auto v : entity)
    {
      auto v_ranks = vertex_ranks.links(v);
      entity_ranks.insert(entity_ranks.end(), v_ranks.begin(), v_ranks.end());
    }
    if (!entity_ranks.empty())
      std::ranges::sort(entity_ranks);

    // If the number of vertices shared with a rank is
    // 'num_vertices_per_e', then add entity data to the send buffer
    auto it = entity_ranks.begin();
    while (it != entity_ranks.end())
    {
      auto it1 = std::find_if(it, entity_ranks.end(),
                              [r0 = *it](auto r1) { return r1 != r0; });
      if (std::ranges::distance(it, it1) == num_vertices_per_e)
      {
        vertex_map.local_to_global(entity, vglobal);
        std::ranges::sort(vglobal);
        entity_to_local_idx.insert(entity_to_local_idx.end(), vglobal.begin(),
                                   vglobal.end());
        entity_to_local_idx.push_back(id);

        // Only send entities that are not known to be ghosts
        if (ghost_status[id] != 1)
        {
          // Entity id may be shared with neighbourhood rank r
          const std::size_t r = *it;
          send_entities[r].insert(send_entities[r].end(), vglobal.begin(),
                                  vglobal.end());
          send_index[r].push_back(id);
        }
      }

      it = it1;
    }
  }
}
//-----------------------------------------------------------------------------

/// @brief Map the entity index of instances `[p0, p1)` through
/// `local_index`.
///
/// This code is thread-safe: calls write only their own range.
///
/// @param[in] p0 First instance to map.
/// @param[in] p1 One past the last instance to map.
/// @param[in] entity_index Entity index of each instance.
/// @param[in] local_index New index of each entity.
/// @param[out] new_entity_index New entity index of each instance.
void renumber_instances(std::size_t p0, std::size_t p1,
                        std::span<const std::int32_t> entity_index,
                        std::span<const std::int32_t> local_index,
                        std::span<std::int32_t> new_entity_index)
{
  for (std::size_t p = p0; p < p1; ++p)
    new_entity_index[p] = local_index[entity_index[p]];
}
//-----------------------------------------------------------------------------

/// Communicate with sharing processes to find out which entities are
/// ghosts and return a map (vector) to move these local indices to the
/// end of the local range. Also returns the index map, and shared
/// entities, i.e. the set of all processes which share each shared
/// entity.
///
/// @param[in] comm MPI Communicator
/// @param[in] vertex_map Index map for vertex distribution
/// @param[in] entity_list List of entities, each entity represented by
/// its local vertex indices
/// @param[in] num_vertices_per_e Number of vertices per entity
/// @param[in] ghost_status Ownership/ghost status of each row in
/// `entity_list`
/// @param[in] entity_index Initial numbering for each row in
/// `entity_list`
/// @param[in] entity_count Number of entities.
/// @param[in] first_instance An instance of each entity, i.e. a row of
/// `entity_list` holding its vertices. All instances of an entity have
/// the same (globally oriented) vertex list, so any one of them will
/// do; the caller has them to hand from the labelling.
/// @param[in] num_threads Number of threads to use.
/// @returns Local indices, the index map and shared entities
std::tuple<std::vector<int>, common::IndexMap, std::vector<std::int32_t>>
get_local_indexing(MPI_Comm comm, const common::IndexMap& vertex_map,
                   std::span<const std::int32_t> entity_list,
                   int num_vertices_per_e,
                   std::span<const std::int8_t> ghost_status,
                   std::span<const std::int32_t> entity_index,
                   std::int32_t entity_count,
                   std::span<const std::int32_t> first_instance,
                   int num_threads)
{
  // entity_list contains all the entities for all the cells,
  // listed as local vertex indices, and entity_index contains
  // the initial numbering of the entities.
  //                   entity_list entity_index
  // e.g. cell0-ent0: [0,1,2]      15
  //      cell0-ent1: [1,2,3]      23
  //      cell1-ent0: [0,1,2]      15
  //      cell1-ent1: [1,2,6]      24
  //      ...

  common::Timer timer_li("Entity local indexing");

  //---------
  // Create a symmetric neighbor_comm from vertex_ranks
  common::Timer timer_li_nc("Entity local indexing: neighbourhood setup");

  // Ranks sharing an owned or ghost vertex with this rank, and for each
  // vertex the sharing ranks as positions in all_ranks
  auto [all_ranks, data, offsets]
      = common::compute_sharing_neighbourhood(vertex_map);
  graph::AdjacencyList<int> vertex_ranks(std::move(data), std::move(offsets));

  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(comm, all_ranks.size(), all_ranks.data(),
                                 MPI_UNWEIGHTED, all_ranks.size(),
                                 all_ranks.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbor_comm);

  timer_li_nc.stop();
  timer_li_nc.flush();

  std::vector<std::vector<std::int64_t>> send_entities(all_ranks.size());
  std::vector<std::vector<std::int32_t>> send_index(all_ranks.size());

  // Get all "possibly shared" entities, based on vertex sharing. Send
  // to other processes, and see if we get the same back.
  common::Timer timer_li_cand("Entity local indexing: build candidates");

  // Map from entity (defined by global vertex indices) to local entity
  // index
  std::vector<std::int64_t> entity_to_local_idx;
  std::vector<std::int32_t> perm;
  {
    // Each thread takes a range of entities and appends to its own
    // buffers; concatenating those in range order gives the same
    // result as one serial pass.
    std::vector<std::vector<std::int64_t>> e2l_t(num_threads);
    std::vector<std::vector<std::vector<std::int64_t>>> send_entities_t(
        num_threads, std::vector<std::vector<std::int64_t>>(all_ranks.size()));
    std::vector<std::vector<std::vector<std::int32_t>>> send_index_t(
        num_threads, std::vector<std::vector<std::int32_t>>(all_ranks.size()));
    {
      std::vector<std::jthread> threads;
      for (int i = 1; i < num_threads; ++i)
      {
        auto [e0, e1] = common::local_range(i, entity_count, num_threads);
        threads.emplace_back(
            build_candidates, e0, e1, first_instance, entity_list,
            num_vertices_per_e, std::cref(vertex_ranks), std::cref(vertex_map),
            ghost_status, std::ref(e2l_t[i]), std::ref(send_entities_t[i]),
            std::ref(send_index_t[i]));
      }
      auto [e0, e1] = common::local_range(0, entity_count, num_threads);
      build_candidates(e0, e1, first_instance, entity_list, num_vertices_per_e,
                       vertex_ranks, vertex_map, ghost_status, e2l_t[0],
                       send_entities_t[0], send_index_t[0]);
    }

    for (int i = 0; i < num_threads; ++i)
    {
      entity_to_local_idx.insert(entity_to_local_idx.end(), e2l_t[i].begin(),
                                 e2l_t[i].end());
      for (std::size_t r = 0; r < all_ranks.size(); ++r)
      {
        send_entities[r].insert(send_entities[r].end(),
                                send_entities_t[i][r].begin(),
                                send_entities_t[i][r].end());
        send_index[r].insert(send_index[r].end(), send_index_t[i][r].begin(),
                             send_index_t[i][r].end());
      }
    }

    // entity_to_local_idx rows are [vglobal..., id]; id depends only on
    // vglobal, so excluding it from the sort key and uniqueness check
    // below is safe. That lets a radix sort_by_perm on the leading
    // num_vertices_per_e columns replace the previous, more costly
    // generic lexicographical sort.
    perm = dolfinx::sort_by_perm<std::int64_t>(
        std::span<const std::int64_t>(entity_to_local_idx),
        num_vertices_per_e + 1, num_vertices_per_e);

    auto range_by_key = [&entity_to_local_idx, shape1 = num_vertices_per_e + 1,
                         ncols = num_vertices_per_e](auto e)
    {
      auto begin = std::next(entity_to_local_idx.begin(), e * shape1);
      return std::ranges::subrange(begin, std::next(begin, ncols));
    };

    perm.erase(
        std::ranges::unique(perm, std::ranges::equal, range_by_key).begin(),
        perm.end());
  }

  timer_li_cand.stop();
  timer_li_cand.flush();

  // Get shared entities of this dimension, and also match up an index
  // for the received entities (from other processes) with the indices
  // of the sent entities (to other processes)
  common::Timer timer_li_ex("Entity local indexing: candidate exchange");

  // Send/receive entities
  std::vector<std::int64_t> recv_data;
  std::vector<int> send_sizes, send_disp, recv_disp, recv_sizes;
  {
    std::vector<std::int64_t> send_buffer;
    for (const std::vector<std::int64_t>& x : send_entities)
    {
      send_sizes.push_back(x.size());
      send_buffer.insert(send_buffer.end(), x.begin(), x.end());
    }
    assert(send_sizes.size() == all_ranks.size());

    // Build send displacements
    send_disp = {0};
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::back_inserter(send_disp));

    recv_sizes.resize(all_ranks.size());
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

  // Compare received and sent entity keys. Any received entities not
  // found in entity_to_local_idx will have recv_index set to -1.
  const int mpi_rank = dolfinx::MPI::rank(comm);
  std::vector<std::int32_t> recv_index;
  recv_index.reserve(recv_disp.back() / num_vertices_per_e);
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
          shared_entities_data.push_back({idx, all_ranks[r]});
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

  std::ranges::sort(shared_entities_data);
  const graph::AdjacencyList<int> shared_entities
      = create_adj_list(shared_entities_data, entity_count);

  std::ranges::sort(shared_entity_to_global_vertices_data);
  const graph::AdjacencyList<int> shared_entities_v
      = create_adj_list(shared_entity_to_global_vertices_data, entity_count);

  timer_li_ex.stop();
  timer_li_ex.flush();

  //---------
  // Determine ownership of shared entities
  common::Timer timer_li_own("Entity local indexing: ownership");

  std::vector<std::int32_t> local_index(entity_count, -1);
  std::vector<std::int32_t> interprocess_entities;
  std::int32_t num_local;
  {
    // Index non-ghost entities
    std::int32_t c = 0;
    for (int i = 0; i < entity_count; ++i)
    {
      // Definitely ghost
      if (ghost_status[i] == 1)
        continue;

      if (auto shared_ranks = shared_entities.links(i); shared_ranks.empty())
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
        int owner_rank = get_ownership(shared_ranks, vertices);
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

  timer_li_own.stop();
  timer_li_own.flush();

  //---------
  // Communicate global indices to other processes
  common::Timer timer_li_gi("Entity local indexing: global index exchange");
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
    {
      std::ranges::transform(*x, x->begin(), [num_vertices_per_e](auto a)
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
          std::int32_t p = local_index[idx] - num_local;
          ghost_indices[p] = gi;
          ghost_owners[p] = all_ranks[r];
        }
      }
    }
    assert(std::find(ghost_indices.begin(), ghost_indices.end(), -1)
           == ghost_indices.end());
  }

  timer_li_gi.stop();
  timer_li_gi.flush();

  // Create map from initial numbering to new local indices
  common::Timer timer_li_rn("Entity local indexing: renumber");
  std::vector<std::int32_t> new_entity_index(entity_index.size());
  {
    std::vector<std::jthread> threads;
    for (int i = 1; i < num_threads; ++i)
    {
      auto [p0, p1] = common::local_range(i, entity_index.size(), num_threads);
      threads.emplace_back(renumber_instances, p0, p1, entity_index,
                           std::span<const std::int32_t>(local_index),
                           std::span<std::int32_t>(new_entity_index));
    }
    auto [p0, p1] = common::local_range(0, entity_index.size(), num_threads);
    renumber_instances(p0, p1, entity_index, local_index, new_entity_index);
  }
  timer_li_rn.stop();
  timer_li_rn.flush();

  common::IndexMap index_map(comm, num_local, ghost_indices, ghost_owners);
  return {std::move(new_entity_index), std::move(index_map),
          std::move(interprocess_entities)};
}
//-----------------------------------------------------------------------------

/// @brief Label the entity instances at positions `[p0, p1)` of
/// `sort_order`, numbering the entities that start there from zero.
///
/// Instances of one entity share a key and are therefore consecutive
/// in `sort_order`. A run of instances that begins before `p0` is
/// labelled -1, one less than the first entity labelled here, so that
/// ::shift_labels maps it onto the label the preceding range gave the
/// same run.
///
/// This code is thread-safe. `sort_order` is a permutation, so a call
/// writes only the `entity_index` entries its own range of
/// `sort_order` points at.
///
/// @param[in] sort_order Entity instances, ordered by vertex key.
/// @param[in] p0 First position in `sort_order` to label.
/// @param[in] p1 One past the last position in `sort_order` to label.
/// @param[in] keys Sorted vertex key of each instance, stored
/// column-major (one span per vertex).
/// @param[out] entity_index Entity label of each instance.
/// @param[out] representatives First instance of each entity labelled,
/// in label order. Collected here, rather than by a later pass over
/// `sort_order`, because this loop already has the label of each
/// instance to hand -- recovering it afterwards costs a random read
/// per instance (see ::build_entity_vertices).
void label_entities(std::span<const std::int32_t> sort_order, std::size_t p0,
                    std::size_t p1,
                    std::span<const std::span<std::int32_t>> keys,
                    std::span<std::int32_t> entity_index,
                    std::vector<std::int32_t>& representatives)
{
  auto same_key = [keys](std::int32_t i0, std::int32_t i1)
  {
    for (std::span<const std::int32_t> key : keys)
    {
      if (key[i0] != key[i1])
        return false;
    }
    return true;
  };

  representatives.clear();
  std::size_t p = p0;

  // A run that started before p0 is labelled -1, and its
  // representative left to the range that started it
  if (p0 > 0 and p0 < p1 and same_key(sort_order[p0 - 1], sort_order[p0]))
  {
    std::int32_t idx0 = sort_order[p];
    while (p < p1 and same_key(idx0, sort_order[p]))
      entity_index[sort_order[p++]] = -1;
  }

  std::int32_t label = -1;
  while (p < p1)
  {
    std::int32_t idx0 = sort_order[p];
    representatives.push_back(idx0);
    ++label;
    while (p < p1 and same_key(idx0, sort_order[p]))
      entity_index[sort_order[p++]] = label;
  }
}
//-----------------------------------------------------------------------------

/// @brief Shift the labels ::label_entities wrote at positions
/// `[p0, p1)` of `sort_order` onto the global entity numbering.
///
/// Thread-safe for the same reason as ::label_entities.
///
/// @param[in] sort_order Entity instances, ordered by vertex key.
/// @param[in] p0 First position in `sort_order` to shift.
/// @param[in] p1 One past the last position in `sort_order` to shift.
/// @param[in] offset Number of entities labelled by preceding ranges.
/// @param[in,out] entity_index Entity label of each instance.
void shift_labels(std::span<const std::int32_t> sort_order, std::size_t p0,
                  std::size_t p1, std::int32_t offset,
                  std::span<std::int32_t> entity_index)
{
  for (std::size_t p = p0; p < p1; ++p)
    entity_index[sort_order[p]] += offset;
}
//-----------------------------------------------------------------------------

/// @brief Build the entity-vertex connectivity rows of the entities
/// whose first instances are `representatives`.
///
/// Every instance of an entity holds the same, globally oriented,
/// vertex list, so a row is built from the first instance of its
/// entity (see ::label_entities) and the entity's remaining instances
/// are skipped.
///
/// This code is thread-safe: each entity appears in exactly one
/// `representatives` list, and so is written once.
///
/// @param[in] representatives First instance of each entity to build.
/// @param[in] local_index Local entity index of each instance.
/// @param[in] entity_list Vertices of each instance, flattened.
/// @param[in] num_vertices_per_entity Number of vertices per entity.
/// @param[out] ev Entity-vertex connectivity, flattened, indexed by
/// local entity index.
void build_entity_vertices(std::span<const std::int32_t> representatives,
                           std::span<const std::int32_t> local_index,
                           std::span<const std::int32_t> entity_list,
                           int num_vertices_per_entity,
                           std::span<std::int32_t> ev)
{
  for (std::int32_t idx : representatives)
  {
    std::copy_n(
        std::next(entity_list.begin(), idx * num_vertices_per_entity),
        num_vertices_per_entity,
        std::next(ev.begin(), local_index[idx] * num_vertices_per_entity));
  }
}
//-----------------------------------------------------------------------------

/// Compute entities of dimension d
///
/// @param[in] comm Full topology communicator.
/// @param[in] cell_lists For each cell type: cell-vertex connectivity
/// (flattened), and the index map for the cell distribution.
/// @param[in] vertex_index_map Index map for the vertex distribution.
/// @param[in] entity_type Type of entity to compute.
/// @param[in] dim Topological dimension of the entities to be computed
/// @param[in] num_threads Number of threads to use.
/// @return Returns the (cell-entity connectivity, entity-vertex
/// connectivity, index map for the entity distribution across
/// processes, shared entities)
std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
           graph::AdjacencyList<std::int32_t>, common::IndexMap,
           std::vector<std::int32_t>>
compute_entities_by_key_matching(
    MPI_Comm comm,
    std::vector<std::tuple<mesh::CellType, std::span<const std::int32_t>,
                           std::reference_wrapper<const common::IndexMap>>>
        cell_lists,
    const common::IndexMap& vertex_index_map, mesh::CellType entity_type,
    int dim, int num_threads)
{
  if (dim == 0)
  {
    throw std::runtime_error("Cannot create vertices for "
                             "topology. Should already exist.");
  }

  assert(cell_dim(entity_type) == dim);
  assert(num_threads > 0);

  // Start timer
  common::Timer timer(std::format("Compute entities of dim = {}", dim));

  std::vector<std::vector<std::int32_t>> cell_type_entities(cell_lists.size());
  std::vector<std::int32_t> cell_type_offsets{0};
  for (std::size_t k = 0; k < cell_lists.size(); ++k)
  {
    mesh::CellType cell_type = std::get<0>(cell_lists[k]);
    for (int e = 0; e < cell_num_entities(cell_type, dim); ++e)
    {
      if (cell_entity_type(cell_type, dim, e) == entity_type)
        cell_type_entities[k].push_back(e);
    }

    std::span<const std::int32_t> cells = std::get<1>(cell_lists[k]);
    std::size_t num_cells = cells.size() / mesh::num_cell_vertices(cell_type);
    cell_type_offsets.push_back(cell_type_offsets.back()
                                + num_cells * cell_type_entities[k].size());
  }

  int num_vertices_per_entity = num_cell_vertices(entity_type);

  // Note: these scratch arrays are allocated without initialisation,
  // via std::unique_ptr rather than std::vector. Every element is
  // written by `build_entity_list` below before it is read, so
  // value-initialising them first is pure cost -- several GB, and
  // seconds, for a mesh with tens of millions of cells. Leaving the
  // pages untouched here also moves their first touch into the
  // threaded loop that fills them, where the faults are taken in
  // parallel and the pages land near the thread that will use them.
  const std::size_t entity_list_size
      = static_cast<std::size_t>(cell_type_offsets.back())
        * num_vertices_per_entity;
  std::unique_ptr<std::int32_t[]> entity_list_storage
      = std::make_unique_for_overwrite<std::int32_t[]>(entity_list_size);
  std::span<std::int32_t> entity_list(entity_list_storage.get(),
                                      entity_list_size);

  // Scratch array used only for sorting and matching entities below
  // (entity_list, not this, carries vertex data forward afterwards).
  // Stored column-major, one span per vertex, so the sort needs no
  // per-column extraction copy and the matching loop below can compare
  // a single column directly instead of via a contiguous row span.
  std::unique_ptr<std::int32_t[]> entity_list_sorted_storage
      = std::make_unique_for_overwrite<std::int32_t[]>(entity_list_size);
  std::vector<std::span<std::int32_t>> entity_list_sorted(
      num_vertices_per_entity);
  for (int col = 0; col < num_vertices_per_entity; ++col)
  {
    entity_list_sorted[col] = std::span<std::int32_t>(
        entity_list_sorted_storage.get() + col * cell_type_offsets.back(),
        cell_type_offsets.back());
  }

  for (std::size_t k = 0; k < cell_lists.size(); ++k)
  {
    // Get indices of desired entities within cell. Usually this will be
    // all entities, but for prism or pyramid facets, we will just pick
    // out triangle or quad facets.

    // Create map from cell vertices to entity vertices
    mesh::CellType cell_type = std::get<0>(cell_lists[k]);
    std::size_t num_vertices_per_cell = num_cell_vertices(cell_type);
    auto e_vertices = get_entity_vertices(cell_type, dim);

    common::Timer t_thread("Threaded part");

    std::span<const std::int32_t> cells = std::get<1>(cell_lists[k]);
    int num_entities_per_cell = cell_type_entities[k].size();
    std::size_t num_cells = cells.size() / num_cell_vertices(cell_type);

    std::vector<std::jthread> threads;
    for (int i = 1; i < num_threads; ++i)
    {
      auto [c0, c1] = common::local_range(i, num_cells, num_threads);
      std::size_t offset
          = cell_type_offsets[k] * num_vertices_per_entity
            + c0 * num_vertices_per_entity * num_entities_per_cell;
      std::size_t count
          = (c1 - c0) * num_vertices_per_entity * num_entities_per_cell;
      std::size_t entity_offset
          = cell_type_offsets[k] + c0 * num_entities_per_cell;
      auto cells_i = cells.subspan(c0 * num_vertices_per_cell,
                                   (c1 - c0) * num_vertices_per_cell);
      threads.emplace_back(
          build_entity_list, std::span(entity_list.data() + offset, count),
          entity_offset,
          std::span<const std::span<std::int32_t>>(entity_list_sorted), cells_i,
          num_vertices_per_cell, std::cref(e_vertices), entity_type,
          std::cref(cell_type_entities[k]), std::cref(vertex_index_map));
    }
    auto [c0, c1] = common::local_range(0, num_cells, num_threads);
    std::size_t offset = cell_type_offsets[k] * num_vertices_per_entity
                         + c0 * num_vertices_per_entity * num_entities_per_cell;
    std::size_t count
        = (c1 - c0) * num_vertices_per_entity * num_entities_per_cell;
    std::size_t entity_offset
        = cell_type_offsets[k] + c0 * num_entities_per_cell;
    auto cells_i = cells.subspan(c0 * num_vertices_per_cell,
                                 (c1 - c0) * num_vertices_per_cell);
    build_entity_list(
        std::span(entity_list.data() + offset, count), entity_offset,
        std::span<const std::span<std::int32_t>>(entity_list_sorted), cells_i,
        num_vertices_per_cell, std::cref(e_vertices), entity_type,
        std::cref(cell_type_entities[k]), std::cref(vertex_index_map));
  }

  // Start numbering entities. Uninitialised for the reason given for
  // the scratch arrays above: `sort_order` is a permutation, so the
  // labelling below writes every entry before it is read.
  std::unique_ptr<std::int32_t[]> entity_index_storage
      = std::make_unique_for_overwrite<std::int32_t[]>(
          cell_type_offsets.back());
  std::span<std::int32_t> entity_index(entity_index_storage.get(),
                                       cell_type_offsets.back());
  std::int32_t entity_count = 0;

  // First instance of each entity, grouped by the thread that
  // labelled it (see ::label_entities)
  std::vector<std::vector<std::int32_t>> representatives(num_threads);
  {
    common::Timer timer_number(
        "Compute entities by key matching: number entities");

    auto sort_threaded
        = [](std::span<const std::span<std::int32_t>> cols, int nthreads)
    {
      std::size_t shape0 = cols.empty() ? 0 : cols.front().size();
      std::vector<std::int32_t> sort_order(shape0, 0);
      std::iota(sort_order.begin(), sort_order.end(), 0);
      boost::sort::sample_sort(
          sort_order.begin(), sort_order.end(),
          [cols](auto f0, auto f1)
          {
            for (std::span<const std::int32_t> col : cols)
            {
              if (col[f0] != col[f1])
                return col[f0] < col[f1];
            }
            return false;
          },
          nthreads);

      return sort_order;
    };

    // Sort the list, so that instances of an entity are consecutive
    const std::vector<std::int32_t> sort_order
        = [num_threads, &entity_list_sorted, &sort_threaded]
    {
      if (num_threads == 1)
      {
        std::vector<std::span<const std::int32_t>> cols(
            entity_list_sorted.begin(), entity_list_sorted.end());
        return dolfinx::sort_by_perm(
            std::span<std::span<const std::int32_t>>(cols));
      }
      else
      {
        return sort_threaded(
            std::span<const std::span<std::int32_t>>(entity_list_sorted),
            num_threads);
      }
    }();

    // Label uniquely. Each thread labels its own range of the sorted
    // order, numbering the entities that start in it from zero, and a
    // second pass shifts each range's labels past the entities found by
    // the ranges before it. Counting the entities per range first and
    // only then labelling would instead need a second pass over the
    // (randomly accessed) keys; the shift pass touches only the labels.
    {
      std::vector<std::jthread> threads;
      for (int i = 1; i < num_threads; ++i)
      {
        auto [p0, p1] = common::local_range(i, sort_order.size(), num_threads);
        threads.emplace_back(
            label_entities, std::span<const std::int32_t>(sort_order), p0, p1,
            std::span<const std::span<std::int32_t>>(entity_list_sorted),
            std::span<std::int32_t>(entity_index),
            std::ref(representatives[i]));
      }
      auto [p0, p1] = common::local_range(0, sort_order.size(), num_threads);
      label_entities(sort_order, p0, p1, entity_list_sorted, entity_index,
                     representatives[0]);
    }

    std::vector<std::int32_t> label_offsets(num_threads + 1, 0);
    std::ranges::transform(representatives, std::next(label_offsets.begin()),
                           [](const std::vector<std::int32_t>& r)
                           { return static_cast<std::int32_t>(r.size()); });
    std::partial_sum(std::next(label_offsets.begin()), label_offsets.end(),
                     std::next(label_offsets.begin()));
    entity_count = label_offsets.back();

    {
      // Range 0 is already numbered from zero, so needs no shift
      std::vector<std::jthread> threads;
      for (int i = 1; i < num_threads; ++i)
      {
        if (label_offsets[i] == 0)
          continue;
        auto [p0, p1] = common::local_range(i, sort_order.size(), num_threads);
        threads.emplace_back(
            shift_labels, std::span<const std::int32_t>(sort_order), p0, p1,
            label_offsets[i], std::span<std::int32_t>(entity_index));
      }
    }
  }

  //---------
  // Set ghost status array values
  // 0 = entities with local ownership or ownership that needs deciding
  // 1 = entities that are only in ghost cells (i.e. definitely not
  // owned)
  //
  // Note: left serial. Instances of an entity are scattered through
  // `entity_index`, so a thread-safe split has to walk `sort_order`
  // instead, which costs a random read per instance and is slower than
  // this loop even at high thread counts.
  std::vector<std::int8_t> ghost_status(entity_count, 1);
  for (std::size_t k = 0; k < cell_lists.size(); ++k)
  {
    // Tag all entities in local cells with 0, leaving entities which
    // only appear in ghost cells tagged.
    const common::IndexMap& cell_map = std::get<2>(cell_lists[k]);
    assert(std::size_t(cell_map.size_local() + cell_map.num_ghosts())
           == std::get<1>(cell_lists[k]).size()
                  / mesh::num_cell_vertices(std::get<0>(cell_lists[k])));
    std::int32_t ghost_offset = cell_map.size_local();
    int num_entities_per_cell = cell_type_entities[k].size();
    std::size_t offset = cell_type_offsets[k];
    for (std::int32_t i = 0; i < ghost_offset * num_entities_per_cell; ++i)
    {
      std::int32_t idx = entity_index[i + offset];
      ghost_status[idx] = 0;
    }
  }

  // Communicate with other processes to find out which entities are
  // ghosted and shared. Remap the numbering so that ghosts are at the
  // end.
  // An instance of each entity, in entity index order: the labelling
  // recorded the first instance of every entity it numbered, and
  // range `i` numbered entities [label_offsets[i], label_offsets[i+1])
  std::vector<std::int32_t> first_instance;
  first_instance.reserve(entity_count);
  for (const std::vector<std::int32_t>& r : representatives)
    first_instance.insert(first_instance.end(), r.begin(), r.end());
  assert(first_instance.size() == static_cast<std::size_t>(entity_count));

  auto [local_index, index_map, interprocess_entities] = get_local_indexing(
      comm, vertex_index_map, entity_list, num_vertices_per_entity,
      ghost_status, entity_index, entity_count, first_instance, num_threads);

  // Entity-vertex connectivity. All instances of an entity hold the
  // same, globally oriented, vertex list, so each row is built once,
  // from the first instance of its entity in `sort_order`
  std::vector<std::int32_t> ev_array(entity_count * num_vertices_per_entity);
  {
    std::vector<std::jthread> threads;
    for (int i = 1; i < num_threads; ++i)
    {
      threads.emplace_back(build_entity_vertices,
                           std::span<const std::int32_t>(representatives[i]),
                           std::span<const std::int32_t>(local_index),
                           std::span<const std::int32_t>(entity_list),
                           num_vertices_per_entity,
                           std::span<std::int32_t>(ev_array));
    }
    build_entity_vertices(representatives[0], local_index, entity_list,
                          num_vertices_per_entity, ev_array);
  }
  graph::AdjacencyList ev = graph::regular_adjacency_list(
      std::move(ev_array), num_vertices_per_entity);

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
///
/// @param[in] c_d0_0 The d0 -> 0 (entity (d0) to vertex) connectivity
/// @param[in] c_d1_0 The d1 -> 0 (entity (d1) to vertex) connectivity
/// @return The d0 -> d1 connectivity
graph::AdjacencyList<std::int32_t>
compute_from_map(const graph::AdjacencyList<std::int32_t>& c_d0_0,
                 const graph::AdjacencyList<std::int32_t>& c_d1_0)
{
  // Map from sorted edge vertices to edge index. Built once and then
  // read-only, so an open-addressed map (no per-element allocation, no
  // pointer-chasing) is strictly faster than a node-based map here.
  boost::unordered_flat_map<std::array<std::int32_t, 2>, std::int32_t>
      edge_to_index;
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
mesh::compute_entities(const Topology& topology, int dim, CellType entity_type,
                       int num_threads)
{
  if (num_threads < 1)
    throw std::invalid_argument("num_threads must be >= 1.");

  spdlog::info("Computing mesh entities of dimension {}", dim);

  // Vertices must always exist
  if (dim == 0)
  {
    return {std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>(),
            nullptr, nullptr, std::vector<std::int32_t>()};
  }

  {
    auto idx = std::ranges::find(topology.entity_types(dim), entity_type);
    if (idx == topology.entity_types(dim).end())
    {
      throw std::invalid_argument(std::format(
          "entity_type is not an entity type of the topology at dim={}.", dim));
    }
    int index = std::ranges::distance(topology.entity_types(dim).begin(), idx);
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
  std::vector<std::tuple<mesh::CellType, std::span<const std::int32_t>,
                         std::reference_wrapper<const common::IndexMap>>>
      cell_lists;

  auto cell_index_maps = topology.index_maps(tdim);
  for (std::size_t i = 0; i < cell_types.size(); ++i)
  {
    auto cell_map = cell_index_maps[i];
    assert(cell_map);
    auto cells = topology.connectivity({tdim, int(i)}, {0, 0});
    if (!cells)
      throw std::runtime_error("Cell connectivity missing.");
    cell_lists.push_back({cell_types[i], cells->array(), *cell_map});
  }

  auto vertex_map = topology.index_map(0);
  assert(vertex_map);

  // c->e, e->v
  auto [d0, d1, im, interprocess_entities] = compute_entities_by_key_matching(
      topology.comm(), cell_lists, *vertex_map, entity_type, dim, num_threads);

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
    throw std::runtime_error(
        std::format("Missing entities of dimension {}.", d0[0]));
  }

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d1_0
      = topology.connectivity(d1, {0, 0});
  if (d1[0] > 0 and !topology.connectivity(d1, {0, 0}))
  {
    throw std::runtime_error(
        std::format("Missing entities of dimension {}.", d1[0]));
  }

  // Start timer
  common::Timer timer(std::format("Compute connectivity {}-{}", d0[0], d1[1]));

  // Decide how to compute the connectivity
  if (d0 == d1)
  {
    return {std::make_shared<graph::AdjacencyList<std::int32_t>>(
                c_d0_0->num_nodes()),
            nullptr};
  }
  else if (d0[0] < d1[0])
  {
    // Compute connectivity d1 - d0 (if needed), and take
    // transpose
    if (!topology.connectivity(d1, d0))
    {
      // Only possible case is edge->facet
      if (d0[0] != 1 or d1[0] != 2)
      {
        throw std::invalid_argument(
            std::format("Cannot compute connectivity ({}, {})-({}, {}): only "
                        "edge-to-facet connectivity can be computed this way.",
                        d0[0], d0[1], d1[0], d1[1]));
      }
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
    if (d0[0] != 2 or d1[0] != 1)
    {
      throw std::invalid_argument(
          std::format("Cannot compute connectivity ({}, {})-({}, {}): only "
                      "facet-to-edge connectivity can be computed this way.",
                      d0[0], d0[1], d1[0], d1[1]));
    }
    auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        compute_from_map(*c_d0_0, *c_d1_0));
    return {c_d0_d1, nullptr};
  }
  else
    throw std::invalid_argument(
        "Entity dimension error when computing topology.");
}
//--------------------------------------------------------------------------
