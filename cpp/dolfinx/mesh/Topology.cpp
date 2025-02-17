// Copyright (C) 2006-2024 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Topology.h"
#include "cell_types.h"
#include "permutationcomputation.h"
#include "topologycomputation.h"
#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <numeric>
#include <random>
#include <set>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
/// @brief Determine owner and sharing ranks sharing an index.
///
/// @note Collective
///
/// Indices are sent to a 'post office' rank, which uses a
/// (deterministic) random number generator to determine which rank is
/// the 'owner'. This information is sent back to the ranks who sent the
/// index to the post office.
///
/// @param[in] comm MPI communicator
/// @param[in] indices Global indices to determine a an owning MPI ranks
/// for.
/// @return Map from global index to sharing ranks for each index in
/// indices. The owner rank is the first as the first in the of ranks.
graph::AdjacencyList<int>
determine_sharing_ranks(MPI_Comm comm, std::span<const std::int64_t> indices)
{
  common::Timer timer("Topology: determine shared index ownership");

  // FIXME: use sensible name
  std::int64_t global_range = 0;
  {
    std::int64_t max_index
        = indices.empty() ? 0 : *std::ranges::max_element(indices);
    MPI_Allreduce(&max_index, &global_range, 1, MPI_INT64_T, MPI_MAX, comm);
    global_range += 1;
  }

  // Build {dest, pos} list, and sort
  std::vector<std::array<int, 2>> dest_to_index;
  {
    const int size = dolfinx::MPI::size(comm);
    dest_to_index.reserve(indices.size());
    for (auto idx : indices)
    {
      int dest = dolfinx::MPI::index_owner(size, idx, global_range);
      dest_to_index.push_back({dest, static_cast<int>(dest_to_index.size())});
    }
    std::ranges::sort(dest_to_index);
  }

  // Build list of neighbour dest ranks and count number of indices to
  // send to each post office
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest0;
  {
    auto it = dest_to_index.begin();
    while (it != dest_to_index.end())
    {
      // Store global rank and find iterator to next global rank
      dest.push_back(it->front());
      auto it1
          = std::find_if(it, dest_to_index.end(),
                         [r = dest.back()](auto& idx) { return idx[0] != r; });

      // Store number of items for current rank
      num_items_per_dest0.push_back(std::distance(it, it1));

      // Advance iterator
      it = it1;
    }
  }

  // Determine src ranks. Sort ranks so that ownership determination is
  // deterministic for a given number of ranks.
  std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  std::ranges::sort(src);

  // Create neighbourhood communicator for sending data to post offices
  MPI_Comm neigh_comm0;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm0);

  // Compute send displacements
  std::vector<std::int32_t> send_disp0(num_items_per_dest0.size() + 1, 0);
  std::partial_sum(num_items_per_dest0.begin(), num_items_per_dest0.end(),
                   std::next(send_disp0.begin()));

  // Send number of items to post offices (destination) that I will be
  // sending
  std::vector<int> num_items_recv0(src.size());
  num_items_per_dest0.reserve(1);
  num_items_recv0.reserve(1);
  MPI_Neighbor_alltoall(num_items_per_dest0.data(), 1, MPI_INT,
                        num_items_recv0.data(), 1, MPI_INT, neigh_comm0);

  // Prepare receive displacement and buffers
  std::vector<std::int32_t> recv_disp0(num_items_recv0.size() + 1, 0);
  std::partial_sum(num_items_recv0.begin(), num_items_recv0.end(),
                   std::next(recv_disp0.begin()));

  // Pack send buffer
  std::vector<int> send_buffer0;
  send_buffer0.reserve(send_disp0.back());
  for (auto idx : dest_to_index)
    send_buffer0.push_back(indices[idx[1]]);

  // Send/receive global indices
  std::vector<int> recv_buffer0(recv_disp0.back());
  MPI_Neighbor_alltoallv(send_buffer0.data(), num_items_per_dest0.data(),
                         send_disp0.data(), MPI_INT, recv_buffer0.data(),
                         num_items_recv0.data(), recv_disp0.data(), MPI_INT,
                         neigh_comm0);
  MPI_Comm_free(&neigh_comm0);

  // -- Transpose

  // Build {global index, pos, src} list
  std::vector<std::array<std::int64_t, 3>> indices_list;
  for (std::size_t p = 0; p < recv_disp0.size() - 1; ++p)
    for (std::int32_t i = recv_disp0[p]; i < recv_disp0[p + 1]; ++i)
      indices_list.push_back({recv_buffer0[i], i, int(p)});
  std::ranges::sort(indices_list);

  // Find which ranks have each index
  std::vector<std::int32_t> num_items_per_dest1(recv_disp0.size() - 1, 0);
  std::vector<std::int32_t> num_items_per_pos1(recv_disp0.back(), 0);

  std::vector<int> owner;
  std::vector<int> disp1 = {0};
  {
    std::mt19937 rng(dolfinx::MPI::rank(comm));
    auto it = indices_list.begin();
    while (it != indices_list.end())
    {
      // Find iterator to next different global index
      auto it1
          = std::find_if(it, indices_list.end(), [idx0 = (*it)[0]](auto& idx)
                         { return idx[0] != idx0; });

      // Number of times index is repeated
      std::size_t num = std::distance(it, it1);

      // Pick an owner
      auto it_owner = it;
      if (num > 1)
      {
        std::uniform_int_distribution<int> distrib(0, num - 1);
        it_owner = std::next(it, distrib(rng));
      }
      owner.push_back(it_owner->at(2));

      // Update number of items to be sent to each rank and record owner
      for (auto itx = it; itx != it1; ++itx)
      {
        const std::array<std::int64_t, 3>& data = *itx;
        num_items_per_pos1[data[1]] = num + 1;
        num_items_per_dest1[data[2]] += num + 1;
      }

      disp1.push_back(disp1.back() + num);

      // Advance iterator
      it = it1;
    }
  }

  // Compute send displacement
  std::vector<std::int32_t> send_disp1(num_items_per_dest1.size() + 1, 0);
  std::partial_sum(num_items_per_dest1.begin(), num_items_per_dest1.end(),
                   std::next(send_disp1.begin()));

  // Build send buffer
  std::vector<int> send_buffer1(send_disp1.back());
  {
    // Compute buffer  displacement
    std::vector<std::int32_t> bdisp1(num_items_per_pos1.size() + 1, 0);
    std::partial_sum(num_items_per_pos1.begin(), num_items_per_pos1.end(),
                     std::next(bdisp1.begin()));

    for (std::size_t i = 0; i < disp1.size() - 1; ++i)
    {
      // Get data for first occurrence of global index
      std::int32_t owner_rank = owner[i];
      std::int32_t num_sharing_ranks = disp1[i + 1] - disp1[i];

      // For each appearance of the global index the sharing ranks
      auto indices_it0 = std::next(indices_list.begin(), disp1[i]);
      auto indices_it1 = std::next(indices_it0, num_sharing_ranks);
      for (std::int32_t j = disp1[i]; j < disp1[i + 1]; ++j)
      {
        auto& data1 = indices_list[j];
        std::size_t pos = data1[1];
        std::int32_t bufferpos = bdisp1[pos];
        send_buffer1[bufferpos] = num_sharing_ranks;

        // Store indices (global)
        auto it0 = std::next(send_buffer1.begin(), bufferpos + 1);
        std::transform(indices_it0, indices_it1, it0,
                       [&src](auto& x) { return src[x[2]]; });

        auto it1 = std::next(it0, num_sharing_ranks);
        auto it_owner = std::find(it0, it1, src[owner_rank]);
        assert(it_owner != it1);
        std::iter_swap(it0, it_owner);
      }
    }
  }

  // Send back
  MPI_Comm neigh_comm1;
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm1);

  // Send number of values to receive
  std::vector<int> num_items_recv1(dest.size());
  num_items_per_dest1.reserve(1);
  num_items_recv1.reserve(1);
  MPI_Neighbor_alltoall(num_items_per_dest1.data(), 1, MPI_INT,
                        num_items_recv1.data(), 1, MPI_INT, neigh_comm1);

  // Prepare receive displacements
  std::vector<std::int32_t> recv_disp1(num_items_recv1.size() + 1, 0);
  std::partial_sum(num_items_recv1.begin(), num_items_recv1.end(),
                   std::next(recv_disp1.begin()));

  // Send data
  std::vector<int> recv_buffer1(recv_disp1.back());
  MPI_Neighbor_alltoallv(send_buffer1.data(), num_items_per_dest1.data(),
                         send_disp1.data(), MPI_INT, recv_buffer1.data(),
                         num_items_recv1.data(), recv_disp1.data(), MPI_INT,
                         neigh_comm1);
  MPI_Comm_free(&neigh_comm1);

  // Build adjacency list
  std::vector<int> data;
  std::vector<std::int32_t> graph_offsets = {0};
  {
    auto it = recv_buffer1.begin();
    while (it != recv_buffer1.end())
    {
      const std::size_t d = std::distance(recv_buffer1.begin(), it);
      std::int64_t num_ranks = *it;

      std::span ranks(recv_buffer1.data() + d + 1, num_ranks);
      data.insert(data.end(), ranks.begin(), ranks.end());
      graph_offsets.push_back(graph_offsets.back() + num_ranks);

      std::advance(it, num_ranks + 1);
    }
  }

  return graph::AdjacencyList(std::move(data), std::move(graph_offsets));
}

/// @brief Build ownership 'groups' (owned/undetermined/non-owned) of
/// vertices.
///
/// Owned vertices are attached only to owned cells and 'unowned'
/// vertices are attached only to ghost cells. Vertices with
/// undetermined ownership are attached to owned and unowned cells.
///
/// @param cells Input owned cells vertices
/// @param cells Input ghost cell vertices
/// @return Sorted lists of vertex indices that are:
/// 1. Owned by the caller
/// 2. With undetermined ownership
/// 3. Not owned by the caller
std::array<std::vector<std::int64_t>, 2> vertex_ownership_groups(
    const std::vector<std::span<const std::int64_t>>& cells_owned,
    const std::vector<std::span<const std::int64_t>>& cells_ghost,
    std::span<const std::int64_t> boundary_vertices)
{
  common::Timer timer("Topology: determine vertex ownership groups (owned, "
                      "undetermined, unowned)");

  // Build set of 'local' cell vertices (attached to an owned cell)
  std::vector<std::int64_t> local_vertex_set;
  local_vertex_set.reserve(
      std::accumulate(cells_owned.begin(), cells_owned.end(), 0,
                      [](std::size_t s, auto& v) { return s + v.size(); }));
  for (auto c : cells_owned)
    local_vertex_set.insert(local_vertex_set.end(), c.begin(), c.end());

  {
    dolfinx::radix_sort(local_vertex_set);
    auto [unique_end, range_end] = std::ranges::unique(local_vertex_set);
    local_vertex_set.erase(unique_end, range_end);
  }
  // Build set of ghost cell vertices (attached to a ghost cell)
  std::vector<std::int64_t> ghost_vertex_set;
  ghost_vertex_set.reserve(
      std::accumulate(cells_ghost.begin(), cells_ghost.end(), 0,
                      [](std::size_t s, auto& v) { return s + v.size(); }));
  for (auto c : cells_ghost)
    ghost_vertex_set.insert(ghost_vertex_set.end(), c.begin(), c.end());

  {
    dolfinx::radix_sort(ghost_vertex_set);
    auto [unique_end, range_end] = std::ranges::unique(ghost_vertex_set);
    ghost_vertex_set.erase(unique_end, range_end);
  }
  // Build difference 1: Vertices attached only to owned cells, and
  // therefore owned by this rank
  std::vector<std::int64_t> owned_vertices;
  std::ranges::set_difference(local_vertex_set, boundary_vertices,
                              std::back_inserter(owned_vertices));

  // Build difference 2: Vertices attached only to ghost cells, and
  // therefore not owned by this rank
  std::vector<std::int64_t> unowned_vertices;
  std::ranges::set_difference(ghost_vertex_set, local_vertex_set,
                              std::back_inserter(unowned_vertices));

  // TODO Check this in debug mode only?
  // Sanity check
  // No vertices in unowned should also be in boundary...
  std::vector<std::int64_t> unowned_vertices_in_error;
  std::ranges::set_intersection(unowned_vertices, boundary_vertices,
                                std::back_inserter(unowned_vertices_in_error));

  if (!unowned_vertices_in_error.empty())
  {
    throw std::runtime_error(
        "Adding boundary vertices in ghost cells not allowed.");
  }

  return {std::move(owned_vertices), std::move(unowned_vertices)};
}
/// @brief Send entity indices for owned entities to processes that
/// share but do not own the entities, and receive index data for
/// entities caller shares but does not own (ghosts).
///
/// @param[in] comm MPI communicator
/// @param[in] indices Vertices on the process boundary and which are
/// numbered by other ranks
/// @param[in] index_to_ranks The sharing ranks for each index in
/// `indices`
/// @param[in] offset The indexing offset for this process, i.e. the
/// global index of local index `idx` is `idx + offset_v`.
/// @param[in] global_indices The input global indices owned by this
/// process
/// @param[in] local_indices The new local index, i.e. for input global
/// index `global_indices[i]` the new local index is `local_indices[i]`.
/// @return Triplets of data for entries in `indices`, with
/// 1. Old global index
/// 2. New global index
/// 3. MPI rank of the owner
std::vector<std::int64_t>
exchange_indexing(MPI_Comm comm, std::span<const std::int64_t> indices,
                  const graph::AdjacencyList<int>& index_to_ranks,
                  std::int64_t offset,
                  std::span<const std::int64_t> global_indices,
                  std::span<const std::int32_t> local_indices)
{
  const int mpi_rank = dolfinx::MPI::rank(comm);

  // Build src and destination ranks
  std::vector<int> src, dest;
  for (std::int32_t i = 0; i < index_to_ranks.num_nodes(); ++i)
  {
    if (auto ranks = index_to_ranks.links(i); ranks.front() == mpi_rank)
      dest.insert(dest.end(), std::next(ranks.begin()), ranks.end());
    else
      src.push_back(ranks.front());
  }

  {
    std::ranges::sort(src);
    auto [unique_end, range_end] = std::ranges::unique(src);
    src.erase(unique_end, range_end);
  }

  {
    std::ranges::sort(dest);
    auto [unique_end, range_end] = std::ranges::unique(dest);
    dest.erase(unique_end, range_end);
  }

  // Pack send data. Use std::vector<std::vector>> since size will be
  // modest (equal to number of neighbour ranks)
  std::vector<std::vector<std::int64_t>> send_buffer(dest.size());
  for (std::int32_t i = 0; i < index_to_ranks.num_nodes(); ++i)
  {
    // Get (global) ranks that share this vertex. Note that first rank
    // is the owner.
    if (auto ranks = index_to_ranks.links(i); ranks.front() == mpi_rank)
    {
      // Get local vertex index
      std::int64_t idx_old = indices[i];
      auto local_it = std::ranges::lower_bound(global_indices, idx_old);
      assert(local_it != global_indices.end() and *local_it == idx_old);
      std::size_t pos = std::distance(global_indices.begin(), local_it);
      std::int64_t idx_new = local_indices[pos] + offset;

      // Owned and shared with these processes (starting from 1, 0 is
      // self)
      for (std::size_t j = 1; j < ranks.size(); ++j)
      {
        // Find rank on the neighborhood comm
        auto it = std::ranges::lower_bound(dest, ranks[j]);
        assert(it != dest.end() and *it == ranks[j]);
        int neighbor = std::distance(dest.begin(), it);

        // Add (old global vertex index, new  global vertex index, owner
        // rank (global))
        send_buffer[neighbor].insert(send_buffer[neighbor].end(),
                                     {idx_old, idx_new, mpi_rank});
      }
    }
  }

  // Send/receive data
  std::vector<std::int64_t> recv_data;
  {
    MPI_Comm comm0;
    MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                   dest.size(), dest.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, false, &comm0);

    // Prepare send sizes and send displacements
    std::vector<int> send_sizes;
    send_sizes.reserve(dest.size());
    std::ranges::transform(send_buffer, std::back_inserter(send_sizes),
                           [](auto& x) { return x.size(); });
    std::vector<int> send_disp(dest.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::next(send_disp.begin()));

    std::vector<std::int64_t> sbuffer;
    sbuffer.reserve(send_disp.back());
    for (const std::vector<std::int64_t>& data : send_buffer)
      sbuffer.insert(sbuffer.end(), data.begin(), data.end());

    // Get receive sizes
    std::vector<int> recv_sizes(src.size());
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, comm0);

    std::vector<int> recv_disp(src.size() + 1, 0);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::next(recv_disp.begin()));
    recv_data = std::vector<std::int64_t>(recv_disp.back());
    MPI_Neighbor_alltoallv(sbuffer.data(), send_sizes.data(), send_disp.data(),
                           MPI_INT64_T, recv_data.data(), recv_sizes.data(),
                           recv_disp.data(), MPI_INT64_T, comm0);

    MPI_Comm_free(&comm0);
  }

  return recv_data;
}

/// @brief Send and receive vertex indices and owning ranks for
/// vertices that lie in the ghost cell region.
///
/// Vertices that are attached to ghost cells but which are not attached
/// to the 'true' boundary between processes may be owned (and therefore
/// numbered) by a rank that does not share any (ghost) cells with the
/// caller. The function is called after all vertices on the true
/// boundary have been numbered, with the vertex indices that still need
/// to be exchanged communicated by the ghost cells.
///
/// @param[in] map0 Map for the entity that has access to all required
/// indices, typically the index map for cells.
/// @param[in] entities0 Vertices of the entities that have access to
/// all required new indices. Indices are 'old' global indices.
/// @param[in] num_entity_vertices Number of vertices per entity.
/// @param[in] nlocal1 Number of owned entities of type '1'.
/// @param[in] offset1 The indexing offset for this process for entities
/// of type '1'. I.e., the global index of local index `idx` is `idx +
/// offset1`.
/// @param[in] global_local_entities1 List of (old global index, new
/// local index) pairs for entities of type '1' that have been numbered.
/// The 'new' global index is `global_local_entities1[i].first +
/// offset1`. For entities that have not yet been assigned a new index,
/// the second entry in the pair is `-1`.
/// @param[in] ghost_owners1 The owning rank for indices that are
/// not owned. If `idx` is the 'new' global index
/// @return List of arrays for each entity, where the entity array contains:
/// 1. Old entity index
/// 2. New global index
/// 3. Rank of the process that owns the entity
std::vector<std::array<std::int64_t, 3>> exchange_ghost_indexing(
    const common::IndexMap& map0, std::span<const std::int64_t> entities0,
    int num_entity_vertices, std::int32_t nlocal1, std::int64_t offset1,
    std::span<const std::pair<std::int64_t, std::int32_t>>
        global_local_entities1,
    std::span<const std::int64_t> ghost_entities1,
    std::span<const int> ghost_owners1)
{
  // Receive index of ghost vertices that are not on the process
  // ('true') boundary from the owner of ghost cells.
  //
  // Note: the ghost cell owner might not be the same as the vertex
  // owner.

  MPI_Comm comm;
  std::span src = map0.src();
  std::span dest = map0.dest();
  MPI_Dist_graph_create_adjacent(map0.comm(), src.size(), src.data(),
                                 MPI_UNWEIGHTED, dest.size(), dest.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  // --

  // For each rank, list of owned vertices that are ghosted by other
  // ranks
  std::vector<std::vector<std::int64_t>> shared_vertices_fwd(dest.size());
  {
    // -- Send cell ghost indices to owner
    MPI_Comm comm1;
    MPI_Dist_graph_create_adjacent(
        map0.comm(), dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(),
        src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);

    // Build list of (owner rank, index) pairs for each ghost index, and
    // sort
    std::vector<std::pair<int, std::int64_t>> owner_to_ghost;
    std::ranges::transform(map0.ghosts(), map0.owners(),
                           std::back_inserter(owner_to_ghost),
                           [](auto idx, auto r) -> std::pair<int, std::int64_t>
                           { return {r, idx}; });
    std::ranges::sort(owner_to_ghost);

    // Build send buffer (the second component of each pair in
    // owner_to_ghost) to send to rank that owns the index
    std::vector<std::int64_t> send_buffer;
    send_buffer.reserve(owner_to_ghost.size());
    std::ranges::transform(owner_to_ghost, std::back_inserter(send_buffer),
                           [](auto x) { return x.second; });

    // Compute send sizes and displacements
    std::vector<int> send_sizes, send_disp{0};
    {
      auto it = owner_to_ghost.begin();
      while (it != owner_to_ghost.end())
      {
        auto it1
            = std::find_if(it, owner_to_ghost.end(),
                           [r = it->first](auto x) { return x.first != r; });
        send_sizes.push_back(std::distance(it, it1));
        send_disp.push_back(send_disp.back() + send_sizes.back());
        it = it1;
      }
    }

    // Exchange number of indices to send/receive from each rank
    std::vector<int> recv_sizes(dest.size(), 0);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, comm1);

    // Prepare receive displacement array
    std::vector<int> recv_disp(dest.size() + 1, 0);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::next(recv_disp.begin()));

    // Send ghost indices to owner, and receive owned indices
    std::vector<std::int64_t> recv_buffer(recv_disp.back());
    MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, recv_buffer.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           comm1);
    MPI_Comm_free(&comm1);

    // Iterate over ranks that ghost cells owned by this rank
    std::array<std::int64_t, 2> local_range = map0.local_range();
    for (std::size_t r = 0; r < recv_disp.size() - 1; ++r)
    {
      assert(r < shared_vertices_fwd.size());
      std::vector<std::int64_t>& shared_vertices = shared_vertices_fwd[r];
      for (int i = recv_disp[r]; i < recv_disp[r + 1]; ++i)
      {
        assert(recv_buffer[i] >= local_range[0]);
        assert(recv_buffer[i] < local_range[1]);
        std::int32_t cell_idx = recv_buffer[i] - local_range[0];
        auto vertices = entities0.subspan(cell_idx * num_entity_vertices,
                                          num_entity_vertices);
        shared_vertices.insert(shared_vertices.end(), vertices.begin(),
                               vertices.end());
      }

      std::ranges::sort(shared_vertices);
      auto [unique_end, range_end] = std::ranges::unique(shared_vertices);
      shared_vertices.erase(unique_end, range_end);
    }
  }

  // Compute send sizes and offsets
  std::vector<int> send_sizes(dest.size());
  std::ranges::transform(shared_vertices_fwd, send_sizes.begin(),
                         [](auto& x) { return 3 * x.size(); });
  std::vector<int> send_disp(dest.size() + 1);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_disp.begin()));

  // Get receive sizes
  std::vector<int> recv_sizes(src.size());
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                        MPI_INT, comm);

  // Pack send buffer
  std::vector<std::int64_t> send_buffer;
  send_buffer.reserve(send_disp.back());
  {
    const int mpi_rank = dolfinx::MPI::rank(comm);

    // Iterate over each rank to send vertex data to
    for (const std::vector<std::int64_t>& vertices_old : shared_vertices_fwd)
    {
      // Iterate over vertex indices (old) for current destination rank
      for (std::int64_t vertex_old : vertices_old)
      {
        // Find new vertex index and determine owning rank
        auto it = std::ranges::lower_bound(
            global_local_entities1,
            std::pair<std::int64_t, std::int32_t>(vertex_old, 0),
            [](auto& a, auto& b) { return a.first < b.first; });
        assert(it != global_local_entities1.end());
        assert(it->first == vertex_old);
        assert(it->second != -1);
        std::int64_t global_idx = it->second < nlocal1
                                      ? it->second + offset1
                                      : ghost_entities1[it->second - nlocal1];
        int owner_rank = it->second < nlocal1
                             ? mpi_rank
                             : ghost_owners1[it->second - nlocal1];

        send_buffer.insert(send_buffer.end(),
                           {vertex_old, global_idx, owner_rank});
      }
    }
  }
  assert(send_buffer.size() == (std::size_t)send_disp.back());

  std::vector<int> recv_disp(src.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_disp.begin()));
  std::vector<std::int64_t> recv_buffer(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                         send_disp.data(), MPI_INT64_T, recv_buffer.data(),
                         recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                         comm);

  std::vector<std::array<std::int64_t, 3>> data;
  data.reserve(recv_buffer.size() / 3);
  for (std::size_t i = 0; i < recv_buffer.size(); i += 3)
    data.push_back({recv_buffer[i], recv_buffer[i + 1], recv_buffer[i + 2]});
  std::ranges::sort(data);
  auto [unique_end, range_end] = std::ranges::unique(data);
  data.erase(unique_end, range_end);

  MPI_Comm_free(&comm);

  return data;
}

/// @brief Convert adjacency list edges from global indexing to local
/// indexing.
///
/// Nodes beyond `num_local_nodes` are discarded.
///
/// @param[in] g Graph with global edge indices
/// @param[in] num_local_nodes Number of nodes to retain in the graph.
/// Typically used to trim ghost nodes.
/// @param[in] global_to_local Sorted array of (global, local) indices.
std::vector<std::int32_t> convert_to_local_indexing(
    std::span<const std::int64_t> g,
    std::span<const std::pair<std::int64_t, std::int32_t>> global_to_local)
{
  std::vector<std::int32_t> data(g.size());
  std::transform(g.begin(), std::next(g.begin(), data.size()), data.begin(),
                 [&global_to_local](auto i)
                 {
                   auto it = std::ranges::lower_bound(
                       global_to_local, i, std::ranges::less(),
                       [](auto& e) { return e.first; });
                   assert(it != global_to_local.end());
                   assert(it->first == i);
                   return it->second;
                 });

  return data;
}
} // namespace

//-----------------------------------------------------------------------------
Topology::Topology(
    std::vector<CellType> cell_types,
    std::shared_ptr<const common::IndexMap> vertex_map,
    std::vector<std::shared_ptr<const common::IndexMap>> cell_maps,
    std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>> cells,
    const std::optional<std::vector<std::vector<std::int64_t>>>& original_index)
    : original_cell_index(original_index
                              ? *original_index
                              : std::vector<std::vector<std::int64_t>>())
{
  assert(!cell_types.empty());
  int tdim = cell_dim(cell_types.front());
#ifndef NDEBUG
  for (auto ct : cell_types)
    assert(cell_dim(ct) == tdim);
#endif

  _entity_types.resize(tdim + 1);
  _entity_types[0] = {mesh::CellType::point};
  _entity_types[tdim] = cell_types;

  // Set data
  _index_maps.insert({{0, 0}, vertex_map});
  _connectivity.insert(
      {{{0, 0}, {0, 0}},
       std::make_shared<graph::AdjacencyList<std::int32_t>>(
           vertex_map->size_local() + vertex_map->num_ghosts())});
  if (tdim > 0)
  {
    for (std::size_t i = 0; i < cell_types.size(); ++i)
    {
      _index_maps.insert({{tdim, (int)i}, cell_maps[i]});
      _connectivity.insert({{{tdim, int(i)}, {0, 0}}, cells[i]});
    }
  }

  // FIXME: This is a hack for setting _interprocess_facets when
  // tdim==1, i.e. the 'facets' are vertices
  if (tdim == 1)
  {
    auto [cell_entity, entity_vertex, index_map, interprocess_entities]
        = compute_entities(*this, 0, CellType::point);
    std::ranges::sort(interprocess_entities);
    _interprocess_facets.push_back(std::move(interprocess_entities));
  }
}
//-----------------------------------------------------------------------------
int Topology::dim() const noexcept
{
  return mesh::cell_dim(_entity_types.back().front());
}
//-----------------------------------------------------------------------------
const std::vector<CellType>& Topology::entity_types(int dim) const
{
  return _entity_types.at(dim);
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const
{
  std::vector<CellType> cell_types = entity_types(dim());
  if (cell_types.size() > 1)
    throw std::runtime_error(
        "Multiple cell types of this dimension. Call cell_types "
        "instead.");
  return cell_types.front();
}
//-----------------------------------------------------------------------------
std::vector<mesh::CellType> Topology::cell_types() const
{
  return entity_types(dim());
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const common::IndexMap>>
Topology::index_maps(int dim) const
{
  std::vector<std::shared_ptr<const common::IndexMap>> maps;
  for (std::size_t i = 0; i < _entity_types[dim].size(); ++i)
  {
    auto it = _index_maps.find({dim, int(i)});
    assert(it != _index_maps.end());
    maps.push_back(it->second);
  }
  return maps;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> Topology::index_map(int dim) const
{
  if (_entity_types[dim].size() > 1)
    throw std::runtime_error(
        "Multiple index maps of this dimension. Call index_maps instead.");
  return this->index_maps(dim).at(0);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(std::array<int, 2> d0, std::array<int, 2> d1) const
{
  if (auto it = _connectivity.find({d0, d1}); it == _connectivity.end())
    return nullptr;
  else
    return it->second;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1) const
{
  return this->connectivity({d0, 0}, {d1, 0});
}
//-----------------------------------------------------------------------------
const std::vector<std::uint32_t>& Topology::get_cell_permutation_info() const
{
  // Check if this process owns or ghosts any cells
  assert(this->index_map(this->dim()));
  if (auto i_map = this->index_map(this->dim());
      _cell_permutations.empty()
      and i_map->size_local() + i_map->num_ghosts() > 0)
  {
    throw std::runtime_error(
        "create_entity_permutations must be called before using this data.");
  }

  return _cell_permutations;
}
//-----------------------------------------------------------------------------
const std::vector<std::uint8_t>& Topology::get_facet_permutations() const
{
  if (auto i_map = this->index_map(this->dim() - 1);
      !i_map
      or (_facet_permutations.empty()
          and i_map->size_local() + i_map->num_ghosts() > 0))
  {
    throw std::runtime_error(
        "create_entity_permutations must be called before using this data.");
  }

  return _facet_permutations;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& Topology::interprocess_facets(int index) const
{
  if (_interprocess_facets.empty())
    throw std::runtime_error("Interprocess facets have not been computed.");
  return _interprocess_facets.at(index);
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& Topology::interprocess_facets() const
{
  return this->interprocess_facets(0);
}
//-----------------------------------------------------------------------------
bool Topology::create_entities(int dim)
{
  // TODO: is this check sufficient/correct? Does not catch the
  // cell_entity entity case. Should there also be a check for
  // connectivity(this->dim(), dim)?

  // Skip if already computed (vertices (dim=0) should always exist)
  if (connectivity(dim, 0))
    return false;

  int tdim = this->dim();
  if (dim == 1 and tdim > 1)
    _entity_types[1] = {mesh::CellType::interval};
  else if (dim == 2 and tdim > 2)
  {
    //  Find all facet types
    std::set<mesh::CellType> e_types;
    for (auto c : _entity_types[tdim])
      for (int i = 0; i < cell_num_entities(c, dim); ++i)
        e_types.insert(cell_facet_type(c, i));
    _entity_types[dim] = std::vector(e_types.begin(), e_types.end());
  }

  // for (std::size_t index = 0; index < this->entity_types(dim).size();
  // ++index)
  for (auto entity = this->entity_types(dim).begin();
       entity != this->entity_types(dim).end(); ++entity)
  {
    int index = std::distance(this->entity_types(dim).begin(), entity);

    // Create local entities
    auto [cell_entity, entity_vertex, index_map, interprocess_entities]
        = compute_entities(*this, dim, *entity);
    for (std::size_t k = 0; k < cell_entity.size(); ++k)
    {
      if (cell_entity[k])
      {
        _connectivity.insert(
            {{{this->dim(), int(k)}, {dim, int(index)}}, cell_entity[k]});
      }
    }

    // TODO: is this check necessary? Seems redundant after the "skip
    // check"
    if (entity_vertex)
      _connectivity.insert({{{dim, int(index)}, {0, 0}}, entity_vertex});

    _index_maps.insert({{dim, int(index)}, index_map});

    // Store interprocess facets
    if (dim == this->dim() - 1)
    {
      std::ranges::sort(interprocess_entities);
      _interprocess_facets.push_back(std::move(interprocess_entities));
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
void Topology::create_connectivity(int d0, int d1)
{
  // Make sure entities exist
  create_entities(d0);
  create_entities(d1);

  // Get the number of different entity types in each dimension
  int num_d0 = this->entity_types(d0).size();
  int num_d1 = this->entity_types(d1).size();

  // Create all connectivities between the two entity dimensions
  for (int i0 = 0; i0 < num_d0; ++i0)
  {
    for (int i1 = 0; i1 < num_d1; ++i1)
    {
      // Compute connectivity
      auto [c_d0_d1, c_d1_d0] = compute_connectivity(*this, {d0, i0}, {d1, i1});

      // NOTE: that to compute the (d0, d1) connections is it sometimes
      // necessary to compute the (d1, d0) connections. We store the (d1,
      // d0) for possible later use, but there is a memory overhead if they
      // are not required. It may be better to not automatically store
      // connectivity that was not requested, but advise in a docstring the
      // most efficient order in which to call this function if several
      // connectivities are needed.

      // TODO: Caching policy/strategy.
      // Concerning the note above: Provide an overload
      // create_connectivity(std::vector<std::pair<int, int>>)?

      // Attach connectivities
      if (c_d0_d1)
        _connectivity.insert({{{d0, i0}, {d1, i1}}, c_d0_d1});

      if (c_d1_d0)
        _connectivity.insert({{{d1, i1}, {d0, i0}}, c_d1_d0});
    }
  }
}
//-----------------------------------------------------------------------------
void Topology::create_entity_permutations()
{
  if (!_cell_permutations.empty())
    return;

  // FIXME: Is creating all entities always required? Could it be made
  // cheaper by doing a local version? This call does quite a lot of
  // parallel work.

  // Create all mesh entities
  int tdim = this->dim();
  for (int d = 0; d < tdim; ++d)
    create_entities(d);

  auto [facet_permutations, cell_permutations]
      = compute_entity_permutations(*this);
  _facet_permutations = std::move(facet_permutations);
  _cell_permutations = std::move(cell_permutations);
}
//-----------------------------------------------------------------------------
MPI_Comm Topology::comm() const
{
  auto it = _index_maps.find({this->dim(), 0});
  assert(it != _index_maps.end());
  return it->second->comm();
}
//-----------------------------------------------------------------------------
Topology mesh::create_topology(
    MPI_Comm comm, const std::vector<CellType>& cell_types,
    std::vector<std::span<const std::int64_t>> cells,
    std::vector<std::span<const std::int64_t>> original_cell_index,
    std::vector<std::span<const int>> ghost_owners,
    std::span<const std::int64_t> boundary_vertices)
{
  common::Timer timer("Topology: create");

  assert(cell_types.size() == cells.size());
  assert(ghost_owners.size() == cells.size());
  assert(original_cell_index.size() == cells.size());

  // Check cell data consistency and compile spans of owned and ghost
  // cells
  spdlog::info("Create topology (generalised)");
  std::vector<std::int32_t> num_local_cells(cell_types.size());
  std::vector<std::span<const std::int64_t>> owned_cells;
  std::vector<std::span<const std::int64_t>> ghost_cells;
  for (std::size_t i = 0; i < cell_types.size(); i++)
  {
    int num_vertices = num_cell_vertices(cell_types[i]);
    if (cells[i].size() % num_vertices != 0)
    {
      throw std::runtime_error("Inconsistent number of cell vertices. Got "
                               + std::to_string(cells[i].size())
                               + ", expected multiple of "
                               + std::to_string(num_vertices) + ".");
    }
    num_local_cells[i] = cells[i].size() / num_vertices;
    num_local_cells[i] -= ghost_owners[i].size();
    owned_cells.push_back(cells[i].first(num_local_cells[i] * num_vertices));
    ghost_cells.push_back(cells[i].last(ghost_owners[i].size() * num_vertices));
  }

  // Create sets of owned and unowned vertices from the cell ownership
  // and the list of boundary vertices
  auto [owned_vertices, unowned_vertices]
      = vertex_ownership_groups(owned_cells, ghost_cells, boundary_vertices);

  // For each vertex whose ownership needs determining, find the sharing
  // ranks. The first index in the list of ranks for a vertex is the
  // owner (as determined by determine_sharing_ranks).
  const graph::AdjacencyList<int> global_vertex_to_ranks
      = determine_sharing_ranks(comm, boundary_vertices);

  // Iterate over vertices that have 'unknown' ownership, and if flagged
  // as owned by determine_sharing_ranks update ownership status
  {
    const int mpi_rank = dolfinx::MPI::rank(comm);
    std::vector<std::int64_t> owned_shared_vertices;
    for (std::size_t i = 0; i < boundary_vertices.size(); ++i)
    {
      // Vertex is shared and owned by this rank if the first sharing
      // rank is my rank
      auto ranks = global_vertex_to_ranks.links(i);
      assert(!ranks.empty());
      if (std::int64_t global_index = boundary_vertices[i];
          ranks.front() == mpi_rank)
      {
        owned_shared_vertices.push_back(global_index);
      }
      else
        unowned_vertices.push_back(global_index);
    }
    dolfinx::radix_sort(unowned_vertices);

    // Add owned but shared vertices to owned_vertices, and sort
    owned_vertices.insert(owned_vertices.end(), owned_shared_vertices.begin(),
                          owned_shared_vertices.end());
    dolfinx::radix_sort(owned_vertices);
  }

  // Number all owned vertices, iterating over vertices cell-wise
  std::vector<std::int32_t> local_vertex_indices(owned_vertices.size(), -1);
  {
    std::int32_t v = 0;
    for (std::size_t i = 0; i < cell_types.size(); ++i)
    {
      for (auto vtx : cells[i])
      {
        if (auto it = std::ranges::lower_bound(owned_vertices, vtx);
            it != owned_vertices.end() and *it == vtx)
        {
          std::size_t pos = std::distance(owned_vertices.begin(), it);
          if (local_vertex_indices[pos] < 0)
            local_vertex_indices[pos] = v++;
        }
      }
    }
  }

  // Compute the global offset for owned (local) vertex indices
  std::int64_t global_offset_v = 0;
  {
    const std::int64_t nlocal = owned_vertices.size();
    MPI_Exscan(&nlocal, &global_offset_v, 1, MPI_INT64_T, MPI_SUM, comm);
  }

  // Get global indices of ghost cells
  std::vector<std::vector<std::int64_t>> cell_ghost_indices;
  std::vector<std::shared_ptr<const common::IndexMap>> index_map_c;
  for (std::size_t i = 0; i < cell_types.size(); ++i)
  {
    std::span cell_idx(original_cell_index[i]);
    cell_ghost_indices.push_back(graph::build::compute_ghost_indices(
        comm, cell_idx.first(num_local_cells[i]),
        cell_idx.last(ghost_owners[i].size()), ghost_owners[i]));

    // Create index maps for each cell type
    index_map_c.push_back(std::make_shared<common::IndexMap>(
        comm, num_local_cells[i], cell_ghost_indices[i], ghost_owners[i],
        static_cast<int>(dolfinx::MPI::tag::consensus_nbx) + i));
  }

  // Send and receive  ((input vertex index) -> (new global index, owner
  // rank)) data with neighbours (for vertices on 'true domain
  // boundary')
  const std::vector<std::int64_t> unowned_vertex_data = exchange_indexing(
      comm, boundary_vertices, global_vertex_to_ranks, global_offset_v,
      owned_vertices, local_vertex_indices);
  assert(unowned_vertex_data.size() % 3 == 0);

  // Unpack received data and build array of ghost vertices and owners
  // of the ghost vertices
  std::vector<std::int64_t> ghost_vertices;
  std::vector<int> ghost_vertex_owners;
  std::vector<std::int32_t> local_vertex_indices_unowned(
      unowned_vertices.size(), -1);
  {
    std::int32_t v = owned_vertices.size();
    for (std::size_t i = 0; i < unowned_vertex_data.size(); i += 3)
    {
      const std::int64_t idx_global = unowned_vertex_data[i];
      auto it = std::ranges::lower_bound(unowned_vertices, idx_global);
      assert(it != unowned_vertices.end() and *it == idx_global);
      std::size_t pos = std::distance(unowned_vertices.begin(), it);
      assert(local_vertex_indices_unowned[pos] < 0);
      local_vertex_indices_unowned[pos] = v++;
      ghost_vertices.push_back(unowned_vertex_data[i + 1]); // New global index
      ghost_vertex_owners.push_back(unowned_vertex_data[i + 2]); // Owning rank
    }

    {
      // TODO: avoid building global_to_local_vertices
      std::vector<std::pair<std::int64_t, std::int32_t>>
          global_to_local_vertices;
      global_to_local_vertices.reserve(owned_vertices.size()
                                       + unowned_vertices.size());
      std::ranges::transform(
          owned_vertices, local_vertex_indices,
          std::back_inserter(global_to_local_vertices), [](auto idx0, auto idx1)
          { return std::pair<std::int64_t, std::int32_t>(idx0, idx1); });
      std::ranges::transform(
          unowned_vertices, local_vertex_indices_unowned,
          std::back_inserter(global_to_local_vertices), [](auto idx0, auto idx1)
          { return std::pair<std::int64_t, std::int32_t>(idx0, idx1); });
      std::ranges::sort(global_to_local_vertices);

      // Send (from the ghost cell owner) and receive global indices for
      // ghost vertices that are not on the process boundary. Data is
      // communicated via ghost cells. Note that the ghost cell owner
      // (who we get the vertex index from) is not necessarily the
      // vertex owner. Repeat for each cell type,
      std::vector<std::array<std::int64_t, 3>> recv_data;
      for (std::size_t i = 0; i < cell_types.size(); ++i)
      {
        int num_cell_vertices = mesh::num_cell_vertices(cell_types[i]);
        std::vector<std::array<std::int64_t, 3>> recv_data_i
            = exchange_ghost_indexing(*index_map_c[i], cells[i],
                                      num_cell_vertices, owned_vertices.size(),
                                      global_offset_v, global_to_local_vertices,
                                      ghost_vertices, ghost_vertex_owners);
        recv_data.insert(recv_data.end(), recv_data_i.begin(),
                         recv_data_i.end());
      }

      // Unpack received data and add to arrays of ghost indices and ghost
      // owners
      for (std::array<std::int64_t, 3>& data : recv_data)
      {
        std::int64_t global_idx_old = data[0];
        auto it0 = std::ranges::lower_bound(unowned_vertices, global_idx_old);
        if (it0 != unowned_vertices.end() and *it0 == global_idx_old)
        {
          if (std::size_t pos = std::distance(unowned_vertices.begin(), it0);
              local_vertex_indices_unowned[pos] < 0)
          {
            local_vertex_indices_unowned[pos] = v++;
            ghost_vertices.push_back(data[1]);
            ghost_vertex_owners.push_back(data[2]);
          }
        }
      }
    }
  }

  // TODO: avoid building global_to_local_vertices

  // Convert input cell topology to local vertex indexing
  std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local_vertices;
  global_to_local_vertices.reserve(owned_vertices.size()
                                   + unowned_vertices.size());
  std::ranges::transform(
      owned_vertices, local_vertex_indices,
      std::back_inserter(global_to_local_vertices),
      [](auto idx0, auto idx1) -> std::pair<std::int64_t, std::int32_t>
      { return {idx0, idx1}; });
  std::ranges::transform(
      unowned_vertices, local_vertex_indices_unowned,
      std::back_inserter(global_to_local_vertices),
      [](auto idx0, auto idx1) -> std::pair<std::int64_t, std::int32_t>
      { return {idx0, idx1}; });
  std::ranges::sort(global_to_local_vertices);

  std::vector<std::vector<std::int32_t>> _cells_local_idx;
  for (std::span<const std::int64_t> c : cells)
  {
    _cells_local_idx.push_back(
        convert_to_local_indexing(c, global_to_local_vertices));
  }

  // -- Create Topology object

  // Determine which ranks ghost vertices that are owned by this rank.
  //
  // Note: Other ranks can ghost vertices that lie inside the 'true'
  // boundary on this process. When we got vertex owner indices via
  // exchange_ghost_indexing, we received data from the ghost cell owner
  // and not necessarily from the vertex owner; therefore, we cannot
  // simply 'transpose' the communication graph to find out who ghosts
  // vertices owned by this rank.
  //
  // TODO: Find a away to get the 'dest' without using
  // compute_graph_edges_nbx. Maybe transpose the
  // exchange_ghost_indexing step, followed by another communication
  // round to the owner?
  //
  // Note: This step is required only for meshes with ghost cells and
  // could be skipped when the mesh is not ghosted.
  std::vector<int> dest;
  {
    // Build list of ranks that own vertices that are ghosted by this
    // rank (out edges)
    std::vector<int> src = ghost_vertex_owners;
    dolfinx::radix_sort(src);
    auto [unique_end, range_end] = std::ranges::unique(src);
    src.erase(unique_end, range_end);
    dest = dolfinx::MPI::compute_graph_edges_nbx(comm, src);
  }

  // Create index map for vertices
  auto index_map_v = std::make_shared<common::IndexMap>(
      comm, owned_vertices.size(), ghost_vertices, ghost_vertex_owners,
      static_cast<int>(dolfinx::MPI::tag::consensus_nbx) + cell_types.size());

  // Set cell index map and connectivity
  std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>> cells_c;
  for (std::size_t i = 0; i < cell_types.size(); ++i)
  {
    cells_c.push_back(std::make_shared<graph::AdjacencyList<std::int32_t>>(
        graph::regular_adjacency_list(std::move(_cells_local_idx[i]),
                                      mesh::num_cell_vertices(cell_types[i]))));
  }

  // Save original cell index
  std::vector<std::vector<std::int64_t>> orig_index;
  std::transform(original_cell_index.begin(), original_cell_index.end(),
                 std::back_inserter(orig_index), [](auto idx)
                 { return std::vector<std::int64_t>(idx.begin(), idx.end()); });

  return Topology(cell_types, index_map_v, index_map_c, cells_c, orig_index);
}
//-----------------------------------------------------------------------------
Topology
mesh::create_topology(MPI_Comm comm, std::span<const std::int64_t> cells,
                      std::span<const std::int64_t> original_cell_index,
                      std::span<const int> ghost_owners, CellType cell_type,
                      std::span<const std::int64_t> boundary_vertices)
{
  spdlog::info("Create topology (single cell type)");
  return create_topology(comm, {cell_type}, {cells}, {original_cell_index},
                         {ghost_owners}, boundary_vertices);
}
//-----------------------------------------------------------------------------
std::tuple<Topology, std::vector<int32_t>, std::vector<int32_t>>
mesh::create_subtopology(const Topology& topology, int dim,
                         std::span<const std::int32_t> entities)
{
  // TODO Call common::get_owned_indices here? Do we want to
  // support `entities` possibly having a ghost on one process that is
  // not in `entities` on the owning process?

  // Create a map from an entity in the sub-topology to the
  // corresponding entity in the topology, and create an index map
  std::shared_ptr<common::IndexMap> submap;
  std::vector<int32_t> subentities;
  {
    // FIXME Make this an input requirement?
    std::vector<std::int32_t> _entities(entities.begin(), entities.end());
    std::ranges::sort(_entities);
    auto [unique_end, range_end] = std::ranges::unique(_entities);
    _entities.erase(unique_end, range_end);

    auto [_submap, _subentities]
        = common::create_sub_index_map(*topology.index_map(dim), _entities);
    submap = std::make_shared<common::IndexMap>(std::move(_submap));
    subentities = std::move(_subentities);
  }

  // Get the vertices in the sub-topology. Use subentities
  // (instead of entities) to ensure vertices for ghost entities are
  // included.

  // Get the vertices in the sub-topology owned by this process
  auto map0 = topology.index_map(0);
  assert(map0);

  // Create map from the vertices in the sub-topology to the vertices in the
  // parent topology, and an index map
  std::shared_ptr<common::IndexMap> submap0;
  std::vector<int32_t> subvertices0;
  {
    std::pair<common::IndexMap, std::vector<int32_t>> map_data
        = common::create_sub_index_map(
            *map0, compute_incident_entities(topology, subentities, dim, 0),
            common::IndexMapOrder::any, true);
    submap0 = std::make_shared<common::IndexMap>(std::move(map_data.first));
    subvertices0 = std::move(map_data.second);
  }

  // Sub-topology entity to vertex connectivity
  const CellType entity_type = cell_entity_type(topology.cell_type(), dim, 0);
  int num_vertices_per_entity = cell_num_entities(entity_type, 0);
  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> sub_e_to_v_vec;
  sub_e_to_v_vec.reserve(subentities.size() * num_vertices_per_entity);
  std::vector<std::int32_t> sub_e_to_v_offsets(1, 0);
  sub_e_to_v_offsets.reserve(subentities.size() + 1);

  // Create vertex-to-subvertex vertex map (i.e. the inverse of
  // subvertex_to_vertex)
  // NOTE: Depending on the sub-topology, this may be densely or sparsely
  // populated. Is a different data structure more appropriate?
  std::vector<std::int32_t> vertex_to_subvertex(
      map0->size_local() + map0->num_ghosts(), -1);
  for (std::size_t i = 0; i < subvertices0.size(); ++i)
    vertex_to_subvertex[subvertices0[i]] = i;

  for (std::int32_t e : subentities)
  {
    for (std::int32_t v : e_to_v->links(e))
    {
      std::int32_t v_sub = vertex_to_subvertex[v];
      assert(v_sub != -1);
      sub_e_to_v_vec.push_back(v_sub);
    }
    sub_e_to_v_offsets.push_back(sub_e_to_v_vec.size());
  }

  auto sub_e_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      std::move(sub_e_to_v_vec), std::move(sub_e_to_v_offsets));

  return {Topology({entity_type}, submap0, {submap}, {sub_e_to_v}),
          std::move(subentities), std::move(subvertices0)};
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
mesh::entities_to_index(const Topology& topology, int dim,
                        std::span<const std::int32_t> entities)
{
  spdlog::info("Build list of mesh entity indices from the entity vertices.");

  // Tagged entity topological dimension
  auto map_e = topology.index_map(dim);
  if (!map_e)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(dim)
                             + "have not been created.");
  }

  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);

  const int num_vertices_per_entity
      = cell_num_entities(cell_entity_type(topology.cell_type(), dim, 0), 0);

  // Build map from ordered local vertex indices (key) to entity index
  // (value)
  std::map<std::vector<std::int32_t>, std::int32_t> entity_key_to_index;
  std::vector<std::int32_t> key(num_vertices_per_entity);
  const int num_entities_mesh = map_e->size_local() + map_e->num_ghosts();
  for (int e = 0; e < num_entities_mesh; ++e)
  {
    auto vertices = e_to_v->links(e);
    std::ranges::copy(vertices, key.begin());
    std::ranges::sort(key);
    auto ins = entity_key_to_index.insert({key, e});
    if (!ins.second)
      throw std::runtime_error("Duplicate mesh entity detected.");
  }

  assert(entities.size() % num_vertices_per_entity == 0);

  // Iterate over all entities and find index
  std::vector<std::int32_t> indices;
  indices.reserve(entities.size() / num_vertices_per_entity);
  std::vector<std::int32_t> vertices(num_vertices_per_entity);
  for (std::size_t e = 0; e < entities.size(); e += num_vertices_per_entity)
  {
    auto v = entities.subspan(e, num_vertices_per_entity);
    std::ranges::copy(v, vertices.begin());
    std::ranges::sort(vertices);
    if (auto it = entity_key_to_index.find(vertices);
        it != entity_key_to_index.end())
    {
      indices.push_back(it->second);
    }
    else
      indices.push_back(-1);
  }

  return indices;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::int32_t>>
mesh::compute_mixed_cell_pairs(const Topology& topology,
                               mesh::CellType facet_type)
{
  int tdim = topology.dim();
  std::vector<mesh::CellType> cell_types = topology.entity_types(tdim);
  std::vector<mesh::CellType> facet_types = topology.entity_types(tdim - 1);

  int facet_index = -1;
  for (std::size_t i = 0; i < facet_types.size(); ++i)
  {
    if (facet_types[i] == facet_type)
    {
      facet_index = i;
      break;
    }
  }
  if (facet_index == -1)
    throw std::runtime_error("Cannot find facet type in topology");

  std::vector<std::vector<std::int32_t>> facet_pair_lists;
  for (std::size_t i = 0; i < cell_types.size(); ++i)
    for (std::size_t j = 0; j < cell_types.size(); ++j)
    {
      std::vector<std::int32_t> facet_pairs_ij;
      auto fci = topology.connectivity({tdim - 1, facet_index},
                                       {tdim, static_cast<int>(i)});
      auto cfi = topology.connectivity({tdim, static_cast<int>(i)},
                                       {tdim - 1, facet_index});

      auto local_facet = [](auto cf, std::int32_t c, std::int32_t f)
      {
        auto it = std::find(cf->links(c).begin(), cf->links(c).end(), f);
        if (it == cf->links(c).end())
          throw std::runtime_error("Bad connectivity");
        return std::distance(cf->links(c).begin(), it);
      };

      if (i == j)
      {
        if (fci)
        {
          for (std::int32_t k = 0; k < fci->num_nodes(); ++k)
          {
            if (fci->num_links(k) == 2)
            {
              std::int32_t c0 = fci->links(k)[0], c1 = fci->links(k)[1];
              facet_pairs_ij.push_back(c0);
              facet_pairs_ij.push_back(local_facet(cfi, c0, k));
              facet_pairs_ij.push_back(c1);
              facet_pairs_ij.push_back(local_facet(cfi, c1, k));
            }
          }
        }
      }
      else
      {
        auto fcj = topology.connectivity({tdim - 1, facet_index},
                                         {tdim, static_cast<int>(j)});
        auto cfj = topology.connectivity({tdim, static_cast<int>(j)},
                                         {tdim - 1, facet_index});
        if (fci and fcj)
        {
          assert(fci->num_nodes() == fcj->num_nodes());
          for (std::int32_t k = 0; k < fci->num_nodes(); ++k)
          {
            if (fci->num_links(k) == 1 and fcj->num_links(k) == 1)
            {
              std::int32_t ci = fci->links(k)[0], cj = fcj->links(k)[0];
              facet_pairs_ij.push_back(ci);
              facet_pairs_ij.push_back(local_facet(cfi, ci, k));
              facet_pairs_ij.push_back(cj);
              facet_pairs_ij.push_back(local_facet(cfj, cj, k));
            }
          }
        }
      }
      facet_pair_lists.push_back(facet_pairs_ij);
    }

  return facet_pair_lists;
}
