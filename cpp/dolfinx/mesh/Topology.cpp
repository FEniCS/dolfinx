// Copyright (C) 2006-2022 Anders Logg and Garth N. Wells
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

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{

//-----------------------------------------------------------------------------
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

  const int size = dolfinx::MPI::size(comm);

  // FIXME: use sensible name
  std::int64_t global_range = 0;
  {
    std::int64_t max_index
        = indices.empty() ? 0
                          : *std::max_element(indices.begin(), indices.end());
    MPI_Allreduce(&max_index, &global_range, 1, MPI_INT64_T, MPI_MAX, comm);
    global_range += 1;
  }

  // Build {dest, pos} list, and sort
  std::vector<std::array<int, 2>> dest_to_index;
  {
    dest_to_index.reserve(indices.size());
    for (auto idx : indices)
    {
      int dest = dolfinx::MPI::index_owner(size, idx, global_range);
      dest_to_index.push_back({dest, static_cast<int>(dest_to_index.size())});
    }
    std::sort(dest_to_index.begin(), dest_to_index.end());
  }

  // Build list of neighbour dest ranks and count number of indices to
  // send to each post office
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest0;
  {
    auto it = dest_to_index.begin();
    while (it != dest_to_index.end())
    {
      // const int neigh_rank = dest.size();

      // Store global rank and find iterator to next global rank
      dest.push_back((*it)[0]);
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
  std::sort(src.begin(), src.end());

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
  {
    for (std::int32_t i = recv_disp0[p]; i < recv_disp0[p + 1]; ++i)
      indices_list.push_back({recv_buffer0[i], i, int(p)});
  }
  std::sort(indices_list.begin(), indices_list.end());

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
      auto it1 = std::find_if(it, indices_list.end(),
                              [idx0 = (*it)[0]](auto& idx)
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
      owner.push_back((*it_owner)[2]);

      // Update number of items to be sent to each rank and record
      // owner
      for (auto itx = it; itx != it1; ++itx)
      {
        auto& data = *itx;
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

  return graph::AdjacencyList<int>(std::move(data), std::move(graph_offsets));
}
//-----------------------------------------------------------------------------

/// @brief Build ownership 'groups' (owned/undetermined/non-owned) of
/// vertices.
///
/// Owned vertices are attached only to owned cells and 'unowned'
/// vertices are attached only to ghost cells. Vertices with
/// undetermined ownership are attached to owned and unowned cells.
///
/// @param cells Input mesh topology
/// @param num_local_cells Number of local (non-ghost) cells. These
/// comes before ghost cells in `cells`.
/// @return Sorted lists of vertex indices that are:
/// 1. Owned by the caller
/// 2. With undetermined ownership
/// 3. Not owned by the caller
std::array<std::vector<std::int64_t>, 2>
vertex_ownership_groups(const graph::AdjacencyList<std::int64_t>& cells,
                        int num_local_cells,
                        std::span<const std::int64_t> boundary_vertices)
{
  common::Timer timer("Topology: determine vertex ownership groups (owned, "
                      "undetermined, unowned)");

  // Build set of 'local' cell vertices (attached to an owned cell)
  std::vector<std::int64_t> local_vertex_set(
      cells.array().begin(),
      std::next(cells.array().begin(), cells.offsets()[num_local_cells]));
  dolfinx::radix_sort(std::span(local_vertex_set));
  local_vertex_set.erase(
      std::unique(local_vertex_set.begin(), local_vertex_set.end()),
      local_vertex_set.end());

  // Build set of ghost cell vertices (attached to a ghost cell)
  std::vector<std::int64_t> ghost_vertex_set(
      std::next(cells.array().begin(), cells.offsets()[num_local_cells]),
      cells.array().end());
  dolfinx::radix_sort(std::span(ghost_vertex_set));
  ghost_vertex_set.erase(
      std::unique(ghost_vertex_set.begin(), ghost_vertex_set.end()),
      ghost_vertex_set.end());

  // Build difference 1: Vertices attached only to owned cells, and
  // therefore owned by this rank
  std::vector<std::int64_t> owned_vertices;
  std::set_difference(local_vertex_set.begin(), local_vertex_set.end(),
                      boundary_vertices.begin(), boundary_vertices.end(),
                      std::back_inserter(owned_vertices));

  // Build difference 2: Vertices attached only to ghost cells, and
  // therefore not owned by this rank
  std::vector<std::int64_t> unowned_vertices;
  std::set_difference(ghost_vertex_set.begin(), ghost_vertex_set.end(),
                      local_vertex_set.begin(), local_vertex_set.end(),
                      std::back_inserter(unowned_vertices));

  // TODO Check this in debug mode only?
  // Sanity check
  // No vertices in unowned should also be in boundary...
  std::vector<std::int64_t> unowned_vertices_in_error;
  std::set_intersection(unowned_vertices.begin(), unowned_vertices.end(),
                        boundary_vertices.begin(), boundary_vertices.end(),
                        std::back_inserter(unowned_vertices_in_error));

  if (unowned_vertices_in_error.size() > 0)
    throw std::runtime_error(
        "Adding boundary vertices in ghost cells not allowed.");

  return {std::move(owned_vertices), std::move(unowned_vertices)};
}
//-----------------------------------------------------------------------------

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
    auto ranks = index_to_ranks.links(i);
    if (ranks.front() == mpi_rank)
      dest.insert(dest.end(), std::next(ranks.begin()), ranks.end());
    else
      src.push_back(ranks.front());
  }
  std::sort(src.begin(), src.end());
  src.erase(std::unique(src.begin(), src.end()), src.end());
  std::sort(dest.begin(), dest.end());
  dest.erase(std::unique(dest.begin(), dest.end()), dest.end());

  // Pack send data. Use std::vector<std::vector>> since size will be
  // modest (equal to number of neighbour ranks)
  std::vector<std::vector<std::int64_t>> send_buffer(dest.size());
  for (std::int32_t i = 0; i < index_to_ranks.num_nodes(); ++i)
  {
    // Get (global) ranks that share this vertex. Note that first rank
    // is the owner.
    auto ranks = index_to_ranks.links(i);
    if (ranks.front() == mpi_rank)
    {
      // Get local vertex index
      std::int64_t idx_old = indices[i];
      auto local_it = std::lower_bound(global_indices.begin(),
                                       global_indices.end(), idx_old);
      assert(local_it != global_indices.end() and *local_it == idx_old);
      std::size_t pos = std::distance(global_indices.begin(), local_it);
      std::int64_t idx_new = local_indices[pos] + offset;

      // Owned and shared with these processes (starting from 1, 0 is
      // self)
      for (std::size_t j = 1; j < ranks.size(); ++j)
      {
        // Find rank on the neighborhood comm
        auto it = std::lower_bound(dest.begin(), dest.end(), ranks[j]);
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
    std::transform(send_buffer.begin(), send_buffer.end(),
                   std::back_inserter(send_sizes),
                   [](auto& x) { return x.size(); });
    std::vector<int> send_disp(dest.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::next(send_disp.begin()));

    std::vector<std::int64_t> sbuffer;
    sbuffer.reserve(send_disp.back());
    for (auto& data : send_buffer)
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
//---------------------------------------------------------------------

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
std::vector<std::array<std::int64_t, 3>>
exchange_ghost_indexing(const common::IndexMap& map0,
                        const graph::AdjacencyList<std::int64_t>& entities0,
                        int nlocal1, std::int64_t offset1,
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
  const std::vector<int>& src = map0.src();
  const std::vector<int>& dest = map0.dest();
  MPI_Dist_graph_create_adjacent(map0.comm(), src.size(), src.data(),
                                 MPI_UNWEIGHTED, dest.size(), dest.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  // --

  // For each rank, list of owned vertices that are ghosted by other ranks
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
    std::transform(map0.ghosts().begin(), map0.ghosts().end(),
                   map0.owners().begin(), std::back_inserter(owner_to_ghost),
                   [](auto idx, auto r) -> std::pair<int, std::int64_t> {
                     return {r, idx};
                   });
    std::sort(owner_to_ghost.begin(), owner_to_ghost.end());

    // Build send buffer (the second component of each pair in
    // owner_to_ghost) to send to rank that owns the index
    std::vector<std::int64_t> send_buffer;
    send_buffer.reserve(owner_to_ghost.size());
    std::transform(owner_to_ghost.begin(), owner_to_ghost.end(),
                   std::back_inserter(send_buffer),
                   [](auto x) { return x.second; });

    // Compute send sizes and displacements
    std::vector<int> send_sizes, send_disp{0};
    auto it = owner_to_ghost.begin();
    while (it != owner_to_ghost.end())
    {
      auto it1 = std::find_if(it, owner_to_ghost.end(),
                              [r = it->first](auto x) { return x.first != r; });
      send_sizes.push_back(std::distance(it, it1));
      send_disp.push_back(send_disp.back() + send_sizes.back());
      it = it1;
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
    auto local_range = map0.local_range();
    for (std::size_t r = 0; r < recv_disp.size() - 1; ++r)
    {
      assert(r < shared_vertices_fwd.size());
      std::vector<std::int64_t>& shared_vertices = shared_vertices_fwd[r];
      for (int i = recv_disp[r]; i < recv_disp[r + 1]; ++i)
      {
        assert(recv_buffer[i] >= local_range[0]);
        assert(recv_buffer[i] < local_range[1]);
        std::int32_t cell_idx = recv_buffer[i] - local_range[0];

        auto vertices = entities0.links(cell_idx);
        shared_vertices.insert(shared_vertices.end(), vertices.begin(),
                               vertices.end());
      }

      std::sort(shared_vertices.begin(), shared_vertices.end());
      shared_vertices.erase(
          std::unique(shared_vertices.begin(), shared_vertices.end()),
          shared_vertices.end());
    }
  }

  // Compute send sizes and offsets
  std::vector<int> send_sizes(dest.size());
  std::transform(shared_vertices_fwd.begin(), shared_vertices_fwd.end(),
                 send_sizes.begin(), [](auto& x) { return 3 * x.size(); });
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
    for (const auto& vertices_old : shared_vertices_fwd)
    {
      // Iterate over vertex indices (old) for current destination rank
      for (auto vertex_old : vertices_old)
      {
        // Find new vertex index and determine owning rank
        auto it = std::lower_bound(
            global_local_entities1.begin(), global_local_entities1.end(),
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
  std::sort(data.begin(), data.end());
  data.erase(std::unique(data.begin(), data.end()), data.end());

  MPI_Comm_free(&comm);

  return data;
}
//---------------------------------------------------------------------------------

/// @brief Convert adjacency list edges from global indexing to local
/// indexing.
///
/// Nodes beyond `num_local_nodes` are discarded.
///
/// @param[in] g Graph with global edge indices
/// @param[in] num_local_nodes Number of nodes to retain in the graph.
/// Typically used to trim ghost nodes.
/// @param[in] global_to_local Sorted array of (global, local) indices.
graph::AdjacencyList<std::int32_t> convert_to_local_indexing(
    const graph::AdjacencyList<std::int64_t>& g, std::size_t num_local_nodes,
    std::span<const std::pair<std::int64_t, std::int32_t>> global_to_local)
{
  std::vector<std::int32_t> offsets(
      g.offsets().begin(), std::next(g.offsets().begin(), num_local_nodes + 1));

  std::vector<std::int32_t> data(offsets.back());
  std::transform(g.array().begin(), std::next(g.array().begin(), data.size()),
                 data.begin(),
                 [&global_to_local](auto i)
                 {
                   auto it = std::lower_bound(
                       global_to_local.begin(), global_to_local.end(),
                       std::pair<std::int64_t, std::int32_t>(i, 0),
                       [](auto& a, auto& b) { return a.first < b.first; });
                   assert(it != global_to_local.end());
                   assert(it->first == i);
                   return it->second;
                 });

  return graph::AdjacencyList<std::int32_t>(std::move(data),
                                            std::move(offsets));
}
} // namespace

//-----------------------------------------------------------------------------
Topology::Topology(MPI_Comm comm, CellType type)
    : _comm(comm), _cell_type(type),
      _connectivity(
          cell_dim(type) + 1,
          std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>(
              cell_dim(type) + 1))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Topology::dim() const noexcept { return _connectivity.size() - 1; }
//-----------------------------------------------------------------------------
void Topology::set_index_map(int dim,
                             std::shared_ptr<const common::IndexMap> map)
{
  assert(dim < (int)_index_map.size());
  _index_map[dim] = map;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> Topology::index_map(int dim) const
{
  assert(dim < (int)_index_map.size());
  return _index_map[dim];
}
//-----------------------------------------------------------------------------
std::int32_t Topology::create_entities(int dim)
{
  // TODO: is this check sufficient/correct? Does not catch the cell_entity
  // entity case. Should there also be a check for
  // connectivity(this->dim(), dim) ?
  // Skip if already computed (vertices (dim=0) should always exist)
  if (connectivity(dim, 0))
    return -1;

  // Create local entities
  const auto [cell_entity, entity_vertex, index_map, interprocess_entities]
      = compute_entities(_comm.comm(), *this, dim);

  if (cell_entity)
    set_connectivity(cell_entity, this->dim(), dim);

  // TODO: is this check necessary? Seems redundant after to the "skip check"
  if (entity_vertex)
    set_connectivity(entity_vertex, dim, 0);

  assert(index_map);
  this->set_index_map(dim, index_map);

  // Store boundary facets
  if (dim == this->dim() - 1)
  {
    _interprocess_facets = std::move(interprocess_entities);
    std::sort(_interprocess_facets.begin(), _interprocess_facets.end());
  }

  return index_map->size_local();
}
//-----------------------------------------------------------------------------
void Topology::create_connectivity(int d0, int d1)
{
  // Make sure entities exist
  create_entities(d0);
  create_entities(d1);

  // Compute connectivity
  const auto [c_d0_d1, c_d1_d0] = compute_connectivity(*this, d0, d1);

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
    set_connectivity(c_d0_d1, d0, d1);
  if (c_d1_d0)
    set_connectivity(c_d1_d0, d1, d0);
}
//-----------------------------------------------------------------------------
void Topology::create_entity_permutations()
{
  if (!_cell_permutations.empty())
    return;

  const int tdim = this->dim();

  // FIXME: Is this always required? Could it be made cheaper by doing a
  // local version? This call does quite a lot of parallel work
  // Create all mesh entities

  for (int d = 0; d < tdim; ++d)
    create_entities(d);

  auto [facet_permutations, cell_permutations]
      = compute_entity_permutations(*this);
  _facet_permutations = std::move(facet_permutations);
  _cell_permutations = std::move(cell_permutations);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1) const
{
  assert(d0 < (int)_connectivity.size());
  assert(d1 < (int)_connectivity[d0].size());
  return _connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
void Topology::set_connectivity(
    std::shared_ptr<graph::AdjacencyList<std::int32_t>> c, int d0, int d1)
{
  assert(d0 < (int)_connectivity.size());
  assert(d1 < (int)_connectivity[d0].size());
  _connectivity[d0][d1] = c;
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
const std::vector<std::int32_t>& Topology::interprocess_facets() const
{
  return _interprocess_facets;
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const noexcept { return _cell_type; }
//-----------------------------------------------------------------------------
MPI_Comm Topology::comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
Topology mesh::create_topology(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
    std::span<const std::int64_t> original_cell_index,
    std::span<const int> ghost_owners, const CellType& cell_type,
    std::span<const std::int64_t> boundary_vertices)
{
  common::Timer timer("Topology: create");

  LOG(INFO) << "Create topology";
  if (cells.num_nodes() > 0
      and cells.num_links(0) != num_cell_vertices(cell_type))
  {
    throw std::runtime_error(
        "Inconsistent number of cell vertices. Got "
        + std::to_string(cells.num_links(0)) + ", expected "
        + std::to_string(num_cell_vertices(cell_type)) + ".");
  }

  const std::int32_t num_local_cells = cells.num_nodes() - ghost_owners.size();

  // Create sets of owned and unowned vertices from the cell ownership
  // and the list of boundary vertices
  auto [owned_vertices, unowned_vertices]
      = vertex_ownership_groups(cells, num_local_cells, boundary_vertices);

  // For each vertex whose ownership needs determining, find the sharing
  // ranks. The first index in the list of ranks for a vertex is the owner
  // (as determined by determine_sharing_ranks).
  const graph::AdjacencyList<int> global_vertex_to_ranks
      = determine_sharing_ranks(comm, boundary_vertices);

  // Iterate over vertices that have 'unknown' ownership, and if flagged
  // as owned by determine_sharing_ranks update ownership status
  {
    const int mpi_rank = dolfinx::MPI::rank(comm);
    std::vector<std::int64_t> owned_shared_vertices;
    for (std::size_t i = 0; i < boundary_vertices.size(); ++i)
    {
      // Vertex is shared and owned by this rank if the first sharing rank
      // is my rank
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
    dolfinx::radix_sort(std::span(unowned_vertices));

    // Add owned but shared vertices to owned_vertices, and sort
    owned_vertices.insert(owned_vertices.end(), owned_shared_vertices.begin(),
                          owned_shared_vertices.end());
    dolfinx::radix_sort(std::span(owned_vertices));
  }

  // Number all owned vertices, iterating over vertices cell-wise
  std::vector<std::int32_t> local_vertex_indices(owned_vertices.size(), -1);
  {
    std::int32_t v = 0;
    for (std::int32_t c = 0; c < cells.num_nodes(); ++c)
    {
      for (auto vtx : cells.links(c))
      {
        auto it = std::lower_bound(owned_vertices.begin(), owned_vertices.end(),
                                   vtx);
        if (it != owned_vertices.end() and *it == vtx)
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

  // Create an index map for cells
  std::shared_ptr<common::IndexMap> index_map_c;

  // Get global indices of ghost cells
  std::span cell_idx(original_cell_index);
  const std::vector cell_ghost_indices = graph::build::compute_ghost_indices(
      comm, cell_idx.first(cells.num_nodes() - ghost_owners.size()),
      std::span(original_cell_index).last(ghost_owners.size()), ghost_owners);

  index_map_c = std::make_shared<common::IndexMap>(
      comm, num_local_cells, cell_ghost_indices, ghost_owners);

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

      auto it = std::lower_bound(unowned_vertices.begin(),
                                 unowned_vertices.end(), idx_global);
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
      std::transform(owned_vertices.begin(), owned_vertices.end(),
                     local_vertex_indices.begin(),
                     std::back_inserter(global_to_local_vertices),
                     [](auto idx0, auto idx1) {
                       return std::pair<std::int64_t, std::int32_t>(idx0, idx1);
                     });
      std::transform(unowned_vertices.begin(), unowned_vertices.end(),
                     local_vertex_indices_unowned.begin(),
                     std::back_inserter(global_to_local_vertices),
                     [](auto idx0, auto idx1) {
                       return std::pair<std::int64_t, std::int32_t>(idx0, idx1);
                     });
      std::sort(global_to_local_vertices.begin(),
                global_to_local_vertices.end());

      // Send (from the ghost cell owner) and receive global indices for
      // ghost vertices that are not on the process boundary. Data is
      // communicated via ghost cells. Note that the ghost cell owner
      // (who we get the vertex index from) is not necessarily the
      // vertex owner.
      const std::vector<std::array<std::int64_t, 3>> recv_data
          = exchange_ghost_indexing(*index_map_c, cells, owned_vertices.size(),
                                    global_offset_v, global_to_local_vertices,
                                    ghost_vertices, ghost_vertex_owners);

      // Unpack received data and add to arrays of ghost indices and ghost
      // owners
      for (auto& data : recv_data)
      {
        const std::int64_t global_idx_old = data[0];
        auto it0 = std::lower_bound(unowned_vertices.begin(),
                                    unowned_vertices.end(), global_idx_old);
        if (it0 != unowned_vertices.end() and *it0 == global_idx_old)
        {
          std::size_t pos = std::distance(unowned_vertices.begin(), it0);
          if (local_vertex_indices_unowned[pos] < 0)
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
  std::transform(owned_vertices.begin(), owned_vertices.end(),
                 local_vertex_indices.begin(),
                 std::back_inserter(global_to_local_vertices),
                 [](auto idx0, auto idx1)
                 { return std::pair<std::int64_t, std::int32_t>(idx0, idx1); });
  std::transform(unowned_vertices.begin(), unowned_vertices.end(),
                 local_vertex_indices_unowned.begin(),
                 std::back_inserter(global_to_local_vertices),
                 [](auto idx0, auto idx1)
                 { return std::pair<std::int64_t, std::int32_t>(idx0, idx1); });
  std::sort(global_to_local_vertices.begin(), global_to_local_vertices.end());

  const std::size_t num_local_nodes = cells.num_nodes();
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> cells_local_idx
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          convert_to_local_indexing(cells, num_local_nodes,
                                    global_to_local_vertices));

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
    dolfinx::radix_sort(std::span(src));
    src.erase(std::unique(src.begin(), src.end()), src.end());
    dest = dolfinx::MPI::compute_graph_edges_nbx(comm, src);
  }

  Topology topology(comm, cell_type);
  const int tdim = topology.dim();

  // Create index map for vertices
  auto index_map_v = std::make_shared<common::IndexMap>(
      comm, owned_vertices.size(), ghost_vertices, ghost_vertex_owners);
  auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      index_map_v->size_local() + index_map_v->num_ghosts());

  // Set vertex index map and 'connectivity'
  topology.set_index_map(0, index_map_v);
  topology.set_connectivity(c0, 0, 0);

  // Set cell index map and connectivity
  topology.set_index_map(tdim, index_map_c);
  topology.set_connectivity(cells_local_idx, tdim, 0);

  // Save original cell index
  topology.original_cell_index.assign(
      original_cell_index.begin(),
      std::next(original_cell_index.begin(), num_local_nodes));

  return topology;
}
//-----------------------------------------------------------------------------

std::vector<std::int32_t>
mesh::entities_to_index(const Topology& topology, int dim,
                        const graph::AdjacencyList<std::int32_t>& entities)
{
  LOG(INFO) << "Build list if mesh entity indices from the entity vertices.";

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
    std::copy(vertices.begin(), vertices.end(), key.begin());
    std::sort(key.begin(), key.end());
    auto ins = entity_key_to_index.insert({key, e});
    if (!ins.second)
      throw std::runtime_error("Duplicate mesh entity detected.");
  }

  // Iterate over all entities and find index
  std::vector<std::int32_t> indices;
  indices.reserve(entities.num_nodes());
  std::vector<std::int32_t> vertices(num_vertices_per_entity);
  for (std::int32_t e = 0; e < entities.num_nodes(); ++e)
  {
    auto v = entities.links(e);
    assert(num_vertices_per_entity == entities.num_links(e));
    std::copy(v.begin(), v.end(), vertices.begin());
    std::sort(vertices.begin(), vertices.end());

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