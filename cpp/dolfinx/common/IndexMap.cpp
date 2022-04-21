// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include <algorithm>
#include <dolfinx/common/sort.h>
#include <functional>
#include <numeric>

using namespace dolfinx;
using namespace dolfinx::common;

namespace
{
//-----------------------------------------------------------------------------
/// Compute the owning rank of ghost indices
[[maybe_unused]] std::vector<int>
get_ghost_ranks(MPI_Comm comm, std::int32_t local_size,
                const xtl::span<const std::int64_t>& ghosts)
{
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  std::vector<std::int32_t> local_sizes(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                comm);

  // NOTE: We do not use std::partial_sum here as it narrows std::int64_t to
  // std::int32_t.
  // NOTE: Using std::inclusive_scan is possible, but GCC prior to 9.3.0
  // only includes the parallel version of this algorithm, requiring
  // e.g. Intel TBB.
  std::vector<std::int64_t> all_ranges(mpi_size + 1, 0);
  std::transform(all_ranges.cbegin(), std::prev(all_ranges.cend()),
                 local_sizes.cbegin(), std::next(all_ranges.begin()),
                 std::plus<std::int64_t>());

  // Compute rank of ghost owners
  std::vector<int> ghost_ranks(ghosts.size(), -1);
  std::transform(ghosts.cbegin(), ghosts.cend(), ghost_ranks.begin(),
                 [&all_ranges](auto ghost)
                 {
                   auto it = std::upper_bound(all_ranges.cbegin(),
                                              all_ranges.cend(), ghost);
                   return std::distance(all_ranges.cbegin(), it) - 1;
                 });

  return ghost_ranks;
}
//-----------------------------------------------------------------------------

/// @todo This functions returns with a special ordering that is not
/// documented. Document properly.
///
/// Compute (owned) global indices shared with neighbor processes
///
/// @param[in] comm MPI communicator where the neighborhood sources are
/// the owning ranks of the callers ghosts (comm_ghost_to_owner)
/// @param[in] ghosts Global index of ghosts indices on the caller
/// @param[in] ghost_src_ranks The src rank on @p comm for each ghost on
/// the caller
/// @return  (i) For each neighborhood rank (destination ranks on comm)
/// a list of my (owned) global indices that are ghost on the rank and
/// (ii) displacement vector for each rank.
///
/// Within list of owned global indices, for each rank the order of the
/// indices are such that the order is the same as the ordering ghosts
/// on the receiver for a given rank.
std::tuple<std::vector<std::int64_t>, std::vector<std::int32_t>>
compute_owned_shared(MPI_Comm comm, const xtl::span<const std::int64_t>& ghosts,
                     const xtl::span<const std::int32_t>& ghost_src_ranks)
{
  assert(ghosts.size() == ghost_src_ranks.size());

  // Send global index of my ghost indices to the owning rank

  // Get src/dest global ranks for the neighbourhood : src ranks have
  // ghosts, dest ranks hold the index owner
  const auto [src_ranks, dest_ranks] = dolfinx::MPI::neighbors(comm);

  // Compute number of ghost indices to send to each owning rank
  std::vector<int> out_edges_num(dest_ranks.size(), 0);
  std::for_each(ghost_src_ranks.cbegin(), ghost_src_ranks.cend(),
                [&out_edges_num](auto src_rank) { out_edges_num[src_rank]++; });

  // Send number of my 'ghost indices' to each owner, and receive number
  // of my 'owned indices' that are ghosted on other ranks
  std::vector<int> in_edges_num(src_ranks.size());
  in_edges_num.reserve(1);
  out_edges_num.reserve(1);
  MPI_Neighbor_alltoall(out_edges_num.data(), 1, MPI_INT, in_edges_num.data(),
                        1, MPI_INT, comm);

  // Prepare communication displacements
  std::vector<int> send_disp(dest_ranks.size() + 1, 0);
  std::partial_sum(out_edges_num.begin(), out_edges_num.end(),
                   send_disp.begin() + 1);
  std::vector<int> recv_disp(src_ranks.size() + 1, 0);
  std::partial_sum(in_edges_num.begin(), in_edges_num.end(),
                   recv_disp.begin() + 1);

  // Pack my 'ghost indices' to send the owning rank
  std::vector<std::int64_t> send_indices(send_disp.back());
  {
    std::vector<int> insert_disp = send_disp;
    for (std::size_t i = 0; i < ghosts.size(); ++i)
    {
      const int owner_rank = ghost_src_ranks[i];
      send_indices[insert_disp[owner_rank]] = ghosts[i];
      insert_disp[owner_rank]++;
    }
  }

  // May have repeated shared indices with different processes
  std::vector<std::int64_t> recv_indices(recv_disp.back());
  MPI_Neighbor_alltoallv(send_indices.data(), out_edges_num.data(),
                         send_disp.data(), MPI_INT64_T, recv_indices.data(),
                         in_edges_num.data(), recv_disp.data(), MPI_INT64_T,
                         comm);

  // Return global indices received from each rank that ghost my owned
  // indices, and return how many global indices are received from each
  // neighborhood rank
  return {recv_indices, recv_disp};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::vector<int32_t> dolfinx::common::compute_owned_indices(
    const xtl::span<const std::int32_t>& indices, const IndexMap& map)
{
  // Split indices into those owned by this process and those that are
  // ghosts. `ghost_indices` contains the position of the ghost in
  // map.ghosts()
  std::vector<std::int32_t> owned;
  std::vector<std::int32_t> ghost_indices;
  const int size_local = map.size_local();

  // Get number of owned and ghost indices in indicies list to reserve
  // vectors
  const int num_owned = std::count_if(indices.begin(), indices.end(),
                                      [size_local](std::int32_t index)
                                      { return index < size_local; });
  const int num_ghost = indices.size() - num_owned;
  owned.reserve(num_owned);
  ghost_indices.reserve(num_ghost);
  std::for_each(indices.begin(), indices.end(),
                [&owned, &ghost_indices, size_local](const std::int32_t index)
                {
                  if (index < size_local)
                    owned.push_back(index);
                  else
                    ghost_indices.push_back(index - size_local);
                });

  // Create an AdjacencyList whose nodes are the processes in the
  // neighborhood and the links for a given process are the ghosts
  // (global numbering) in `indices` owned by that process.
  MPI_Comm reverse_comm = map.comm(IndexMap::Direction::reverse);
  std::vector<std::int32_t> dest_ranks
      = dolfinx::MPI::neighbors(reverse_comm)[1];
  const std::vector<std::int32_t>& ghost_owner_rank = map.ghost_owner_rank();
  const std::vector<std::int64_t>& ghosts = map.ghosts();
  std::vector<std::int64_t> ghosts_to_send;
  std::vector<std::int32_t> ghosts_per_proc(dest_ranks.size(), 0);

  // Loop through all destination ranks in the neighborhood
  for (std::size_t dest_rank_index = 0; dest_rank_index < dest_ranks.size();
       ++dest_rank_index)
  {
    // Loop through all ghost indices on this rank
    for (std::int32_t ghost_index : ghost_indices)
    {
      // Check if the ghost is owned by the destination rank. If so, add
      // that ghost so it is sent to the correct process.
      if (ghost_owner_rank[ghost_index] == dest_ranks[dest_rank_index])
      {
        ghosts_to_send.push_back(ghosts[ghost_index]);
        ghosts_per_proc[dest_rank_index]++;
      }
    }
  }
  // Create a list of partial sums of the number of ghosts per process
  // and create the AdjacencyList
  std::vector<int> send_disp(dest_ranks.size() + 1, 0);
  std::partial_sum(ghosts_per_proc.begin(), ghosts_per_proc.end(),
                   std::next(send_disp.begin(), 1));
  const graph::AdjacencyList<std::int64_t> data_out(std::move(ghosts_to_send),
                                                    std::move(send_disp));

  // Communicate ghosts on this process in `indices` back to their owners
  const graph::AdjacencyList<std::int64_t> data_in
      = dolfinx::MPI::neighbor_all_to_all(reverse_comm, data_out);

  // Get the local index from the global indices received from other
  // processes and add to `owned`
  const std::vector<std::int64_t>& global_indices = map.global_indices();
  std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local;
  global_to_local.reserve(global_indices.size());
  for (auto idx : global_indices)
  {
    global_to_local.push_back(
        {idx, static_cast<std::int32_t>(global_to_local.size())});
  }
  std::sort(global_to_local.begin(), global_to_local.end());
  std::transform(
      data_in.array().cbegin(), data_in.array().cend(),
      std::back_inserter(owned),
      [&global_to_local](std::int64_t global_index)
      {
        auto it = std::lower_bound(
            global_to_local.begin(), global_to_local.end(),
            typename decltype(global_to_local)::value_type(global_index, 0),
            [](auto& a, auto& b) { return a.first < b.first; });
        assert(it != global_to_local.end() and it->first == global_index);
        return it->second;
      });

  // Sort `owned` and remove non-unique entries (we could have received
  // the same ghost from multiple other processes)
  dolfinx::radix_sort(xtl::span(owned));
  owned.erase(std::unique(owned.begin(), owned.end()), owned.end());

  return owned;
}
//-----------------------------------------------------------------------------
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
common::stack_index_maps(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  // Compute process offset
  const std::int64_t process_offset = std::accumulate(
      maps.cbegin(), maps.cend(), std::int64_t(0),
      [](std::int64_t c, auto& map) -> std::int64_t
      { return c + map.first.get().local_range()[0] * map.second; });

  // Get local map offset
  std::vector<std::int32_t> local_offset(maps.size() + 1, 0);
  for (std::size_t f = 1; f < local_offset.size(); ++f)
  {
    const std::int32_t local_size = maps[f - 1].first.get().size_local();
    const int bs = maps[f - 1].second;
    local_offset[f] = local_offset[f - 1] + bs * local_size;
  }

  // Pack old and new composite indices for owned entries that are ghost
  // on other ranks
  std::vector<std::int64_t> indices;
  for (std::size_t f = 0; f < maps.size(); ++f)
  {
    const int bs = maps[f].second;
    const std::vector<std::int32_t>& forward_indices
        = maps[f].first.get().scatter_fwd_indices().array();
    const std::int64_t offset = bs * maps[f].first.get().local_range()[0];
    for (std::int32_t local_index : forward_indices)
    {
      for (std::int32_t i = 0; i < bs; ++i)
      {
        // Insert field index, global index, composite global index
        indices.insert(
            indices.end(),
            {static_cast<std::int64_t>(f), bs * local_index + i + offset,
             bs * local_index + i + local_offset[f] + process_offset});
      }
    }
  }

  // Build arrays of incoming and outcoming neighborhood ranks
  std::set<std::int32_t> in_neighbor_set, out_neighbor_set;
  for (auto& map : maps)
  {
    MPI_Comm neighbor_comm = map.first.get().comm(IndexMap::Direction::forward);
    auto [source, dest] = dolfinx::MPI::neighbors(neighbor_comm);
    in_neighbor_set.insert(source.begin(), source.end());
    out_neighbor_set.insert(dest.begin(), dest.end());
  }

  const std::vector<int> in_neighbors(in_neighbor_set.begin(),
                                      in_neighbor_set.end());
  const std::vector<int> out_neighbors(out_neighbor_set.begin(),
                                       out_neighbor_set.end());

  // Create neighborhood communicator
  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(
      maps.at(0).first.get().comm(), in_neighbors.size(), in_neighbors.data(),
      MPI_UNWEIGHTED, out_neighbors.size(), out_neighbors.data(),
      MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);

  // Figure out how much data to receive from each neighbor
  const int num_my_rows = indices.size();
  std::vector<int> num_rows_recv(indegree);
  num_rows_recv.reserve(1);
  MPI_Neighbor_allgather(&num_my_rows, 1, MPI_INT, num_rows_recv.data(), 1,
                         MPI_INT, comm);

  // Compute displacements for data to receive
  std::vector<int> disp(indegree + 1, 0);
  std::partial_sum(num_rows_recv.begin(), num_rows_recv.end(),
                   disp.begin() + 1);

  // Send data to neighbors, and receive data
  std::vector<std::int64_t> data_recv(disp.back());
  MPI_Neighbor_allgatherv(indices.data(), indices.size(), MPI_INT64_T,
                          data_recv.data(), num_rows_recv.data(), disp.data(),
                          MPI_INT64_T, comm);

  // Destroy communicator
  MPI_Comm_free(&comm);

  // Create map (old global index -> new global index) for each field
  std::vector<std::map<int64_t, std::int64_t>> ghost_maps(maps.size());
  for (std::size_t i = 0; i < data_recv.size(); i += 3)
    ghost_maps[data_recv[i]].insert({data_recv[i + 1], data_recv[i + 2]});

  /// Build arrays from old ghost index to composite ghost index for
  /// each field
  std::vector<std::vector<std::int64_t>> ghosts_new(maps.size());
  std::vector<std::vector<int>> ghost_owners_new(maps.size());
  for (std::size_t f = 0; f < maps.size(); ++f)
  {
    const int bs = maps[f].second;
    const std::vector<std::int64_t>& ghosts = maps[f].first.get().ghosts();
    const std::vector<int>& ghost_owners
        = maps[f].first.get().ghost_owner_rank();
    for (std::size_t i = 0; i < ghosts.size(); ++i)
    {
      for (int j = 0; j < bs; ++j)
      {
        auto it = ghost_maps[f].find(bs * ghosts[i] + j);
        assert(it != ghost_maps[f].end());
        ghosts_new[f].push_back(it->second);
        ghost_owners_new[f].push_back(ghost_owners[i]);
      }
    }
  }

  return {process_offset, std::move(local_offset), std::move(ghosts_new),
          std::move(ghost_owners_new)};
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm comm, std::int32_t local_size)
    : _comm(comm), _comm_owner_to_ghost(MPI_COMM_NULL),
      _comm_ghost_to_owner(MPI_COMM_NULL), _displs_recv_fwd(1, 0)
{
  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = local_size;
  MPI_Request request_scan;
  MPI_Iexscan(&local_size_tmp, &offset, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_scan);

  // Send local size to sum reduction to get global size
  MPI_Request request;
  MPI_Iallreduce(&local_size_tmp, &_size_global, 1, MPI_INT64_T, MPI_SUM, comm,
                 &request);

  MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  _local_range = {offset, offset + local_size};

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  // FIXME: Remove need to do this
  // Create communicators with empty neighborhoods
  MPI_Comm comm0, comm1;
  std::vector<int> ranks(0);
  // NOTE: create uniform weights as a workaround to issue
  // https://github.com/pmodels/mpich/issues/5764
  std::vector<int> weights(0);
  MPI_Dist_graph_create_adjacent(comm, ranks.size(), ranks.data(),
                                 weights.data(), ranks.size(), ranks.data(),
                                 weights.data(), MPI_INFO_NULL, false, &comm0);
  MPI_Dist_graph_create_adjacent(comm, ranks.size(), ranks.data(),
                                 weights.data(), ranks.size(), ranks.data(),
                                 weights.data(), MPI_INFO_NULL, false, &comm1);
  _comm_owner_to_ghost = dolfinx::MPI::Comm(comm0, false);
  _comm_ghost_to_owner = dolfinx::MPI::Comm(comm1, false);
  _shared_indices = std::make_unique<graph::AdjacencyList<std::int32_t>>(0);
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm comm, std::int32_t local_size,
                   const xtl::span<const int>& dest_ranks,
                   const xtl::span<const std::int64_t>& ghosts,
                   const xtl::span<const int>& src_ranks)
    : _comm(comm), _comm_owner_to_ghost(MPI_COMM_NULL),
      _comm_ghost_to_owner(MPI_COMM_NULL), _ghosts(ghosts.begin(), ghosts.end())
{
  assert(size_t(ghosts.size()) == src_ranks.size());
  assert(std::equal(src_ranks.begin(), src_ranks.end(),
                    get_ghost_ranks(comm, local_size, _ghosts).begin()));

  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = (std::int64_t)local_size;
  MPI_Request request_scan;
  MPI_Iexscan(&local_size_tmp, &offset, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_scan);

  // Send local size to sum reduction to get global size
  MPI_Request request;
  MPI_Iallreduce(&local_size_tmp, &_size_global, 1, MPI_INT64_T, MPI_SUM, comm,
                 &request);

  // Build set of src ranks for ghosts, i.e. the ranks that own the
  // callers ghosts
  std::vector<std::int32_t> halo_src_ranks(src_ranks.begin(), src_ranks.end());
  std::sort(halo_src_ranks.begin(), halo_src_ranks.end());
  halo_src_ranks.erase(
      std::unique(halo_src_ranks.begin(), halo_src_ranks.end()),
      halo_src_ranks.end());

  // Create communicators with directed edges: (0) owner -> ghost,
  // (1) ghost -> owner
  {
    // NOTE: create uniform weights as a workaround to issue
    // https://github.com/pmodels/mpich/issues/5764
    std::vector<int> src_weights(halo_src_ranks.size(), 1);
    std::vector<int> dest_weights(dest_ranks.size(), 1);

    MPI_Comm comm0;
    MPI_Dist_graph_create_adjacent(
        comm, halo_src_ranks.size(), halo_src_ranks.data(), src_weights.data(),
        dest_ranks.size(), dest_ranks.data(), dest_weights.data(),
        MPI_INFO_NULL, false, &comm0);
    _comm_owner_to_ghost = dolfinx::MPI::Comm(comm0, false);

    MPI_Comm comm1;
    MPI_Dist_graph_create_adjacent(comm, dest_ranks.size(), dest_ranks.data(),
                                   dest_weights.data(), halo_src_ranks.size(),
                                   halo_src_ranks.data(), src_weights.data(),
                                   MPI_INFO_NULL, false, &comm1);
    _comm_ghost_to_owner = dolfinx::MPI::Comm(comm1, false);
  }

  // Map ghost owner rank to the rank on neighborhood communicator
  int myrank = -1;
  MPI_Comm_rank(comm, &myrank);
  assert(std::find(src_ranks.begin(), src_ranks.end(), myrank)
         == src_ranks.end());
  std::vector<std::int32_t> ghost_owners(ghosts.size());
  std::transform(src_ranks.cbegin(), src_ranks.cend(), ghost_owners.begin(),
                 [&halo_src_ranks](auto src)
                 {
                   // Get rank of owner on the neighborhood communicator
                   // (rank of out edge on _comm_owner_to_ghost)
                   auto it = std::find(halo_src_ranks.cbegin(),
                                       halo_src_ranks.cend(), src);
                   assert(it != halo_src_ranks.end());
                   return std::distance(halo_src_ranks.cbegin(), it);
                 });

  // Compute owned indices which are ghosted by other ranks, and how
  // many of my indices each neighbor ghosts
  auto [shared_ind, shared_disp] = compute_owned_shared(
      _comm_ghost_to_owner.comm(), _ghosts, ghost_owners);

  // Wait for MPI_Iexscan to complete (get offset)
  MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  _local_range = {offset, offset + local_size};

  // Convert owned global indices that are ghosts on another rank to
  // local indexing
  std::vector<std::int32_t> local_shared_ind(shared_ind.size());
  std::transform(
      shared_ind.cbegin(), shared_ind.cend(), local_shared_ind.begin(),
      [offset](std::int64_t x) -> std::int32_t { return x - offset; });

  _shared_indices = std::make_unique<graph::AdjacencyList<std::int32_t>>(
      std::move(local_shared_ind), std::move(shared_disp));

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  // --- Prepare send and receive size and displacement arrays for
  // scatters. The data is for a forward (owner data -> ghosts)
  // scatters. The reverse (ghosts -> owner) scatter use the transpose
  // of these array.

  // Get number of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);

  // Create displacement vectors fwd scatter
  _sizes_recv_fwd.resize(indegree, 0);
  std::for_each(ghost_owners.cbegin(), ghost_owners.cend(),
                [&recv = _sizes_recv_fwd](auto owner) { recv[owner] += 1; });

  _displs_recv_fwd.resize(indegree + 1, 0);
  std::partial_sum(_sizes_recv_fwd.begin(), _sizes_recv_fwd.end(),
                   _displs_recv_fwd.begin() + 1);

  const std::vector<int32_t>& displs_send = _shared_indices->offsets();
  _sizes_send_fwd.resize(outdegree, 0);
  std::adjacent_difference(displs_send.begin() + 1, displs_send.end(),
                           _sizes_send_fwd.begin());

  // Build array that maps ghost indicies to a position in the recv
  // (forward scatter) and send (reverse scatter) buffers
  _ghost_pos_recv_fwd.resize(ghost_owners.size());
  std::transform(ghost_owners.cbegin(), ghost_owners.cend(),
                 _ghost_pos_recv_fwd.begin(),
                 [displs = _displs_recv_fwd](auto owner) mutable
                 { return displs[owner]++; });
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMap::local_range() const noexcept
{
  return _local_range;
}
//-----------------------------------------------------------------------------
std::int32_t IndexMap::num_ghosts() const noexcept { return _ghosts.size(); }
//-----------------------------------------------------------------------------
std::int32_t IndexMap::size_local() const noexcept
{
  return _local_range[1] - _local_range[0];
}
//-----------------------------------------------------------------------------
std::int64_t IndexMap::size_global() const noexcept { return _size_global; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& IndexMap::ghosts() const noexcept
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
void IndexMap::local_to_global(const xtl::span<const std::int32_t>& local,
                               const xtl::span<std::int64_t>& global) const
{
  assert(local.size() <= global.size());
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  std::transform(
      local.cbegin(), local.cend(), global.begin(),
      [local_size, local_range = _local_range[0], &ghosts = _ghosts](auto local)
      {
        if (local < local_size)
          return local_range + local;
        else
        {
          assert((local - local_size) < (int)ghosts.size());
          return ghosts[local - local_size];
        }
      });
}
//-----------------------------------------------------------------------------
void IndexMap::global_to_local(const xtl::span<const std::int64_t>& global,
                               const xtl::span<std::int32_t>& local) const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];

  std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts(
      _ghosts.size());
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    global_local_ghosts[i] = {_ghosts[i], i + local_size};
  std::map<std::int64_t, std::int32_t> global_to_local(
      global_local_ghosts.begin(), global_local_ghosts.end());

  std::transform(global.cbegin(), global.cend(), local.begin(),
                 [range = _local_range,
                  &global_to_local](std::int64_t index) -> std::int32_t
                 {
                   if (index >= range[0] and index < range[1])
                     return index - range[0];
                   else
                   {
                     auto it = global_to_local.find(index);
                     return it != global_to_local.end() ? it->second : -1;
                   }
                 });
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> IndexMap::global_indices() const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  const std::int32_t num_ghosts = _ghosts.size();
  const std::int64_t global_offset = _local_range[0];
  std::vector<std::int64_t> global(local_size + num_ghosts);
  std::iota(global.begin(), std::next(global.begin(), local_size),
            global_offset);
  std::copy(_ghosts.cbegin(), _ghosts.cend(),
            std::next(global.begin(), local_size));
  return global;
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>&
IndexMap::scatter_fwd_indices() const noexcept
{
  assert(_shared_indices);
  return *_shared_indices;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>&
IndexMap::scatter_fwd_ghost_positions() const noexcept
{
  return _ghost_pos_recv_fwd;
}
//-----------------------------------------------------------------------------
std::vector<int> IndexMap::ghost_owner_neighbor_rank() const
{
  /// Compute the owner on the neighborhood communicator of ghost indices
  std::vector<int> owners;
  const std::vector<int>& displs_recv_fwd = _displs_recv_fwd;
  std::transform(_ghost_pos_recv_fwd.cbegin(), _ghost_pos_recv_fwd.cend(),
                 std::back_inserter(owners),
                 [&displs_recv_fwd](auto ghost_pos)
                 {
                   auto it
                       = std::upper_bound(displs_recv_fwd.cbegin(),
                                          displs_recv_fwd.cend(), ghost_pos);
                   return std::distance(displs_recv_fwd.cbegin(), it) - 1;
                 });
  return owners;
}
//----------------------------------------------------------------------------
std::vector<int> IndexMap::ghost_owner_rank() const
{
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_owner_to_ghost.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  // Compute index owner on neighbourhood comm
  const std::vector<int> ghost_owners = ghost_owner_neighbor_rank();

  std::vector<std::int32_t> owners(ghost_owners.size());
  std::transform(ghost_owners.cbegin(), ghost_owners.cend(), owners.begin(),
                 [&neighbors_in](auto ghost_owner)
                 { return neighbors_in[ghost_owner]; });

  return owners;
}
//----------------------------------------------------------------------------
MPI_Comm IndexMap::comm() const { return _comm.comm(); }
//----------------------------------------------------------------------------
MPI_Comm IndexMap::comm(Direction dir) const
{
  switch (dir)
  {
  case Direction::reverse:
    return _comm_ghost_to_owner.comm();
  case Direction::forward:
    return _comm_owner_to_ghost.comm();
  default:
    throw std::runtime_error("Unknown edge direction for communicator.");
  }
}
//----------------------------------------------------------------------------
std::map<std::int32_t, std::set<int>> IndexMap::compute_shared_indices() const
{
  // Get number of neighbors and neighbor ranks
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_owner_to_ghost.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  std::map<std::int32_t, std::set<int>> shared_indices;

  // Build map from owned local index to ranks that ghost the index
  for (std::int32_t p = 0; p < _shared_indices->num_nodes(); ++p)
  {
    const int rank_global = neighbors_out[p];
    for (std::int32_t idx : _shared_indices->links(p))
      shared_indices[idx].insert(rank_global);
  }

  // Ghost indices know the owner rank, but they don't know about other
  // ranks that also ghost the index. If an index is a ghost on more
  // than one rank, we need to send each rank that ghosts the index the
  // other ranks which also ghost the index.

  std::vector<std::int64_t> fwd_sharing_data;
  std::vector<int> fwd_sharing_offsets{0};
  for (std::int32_t p = 0; p < _shared_indices->num_nodes(); ++p)
  {
    for (std::int32_t idx : _shared_indices->links(p))
    {
      assert(shared_indices.find(idx) != shared_indices.end());
      if (auto it = shared_indices.find(idx); it->second.size() > 1)
      {
        // Add global index and number of sharing ranks
        fwd_sharing_data.insert(
            fwd_sharing_data.end(),
            {idx + _local_range[0],
             static_cast<std::int64_t>(shared_indices[idx].size())});

        // Add sharing ranks
        fwd_sharing_data.insert(fwd_sharing_data.end(), it->second.begin(),
                                it->second.end());
      }
    }

    // Add process offset
    fwd_sharing_offsets.push_back(fwd_sharing_data.size());
  }

  // Send sharing rank data from owner to ghosts

  // Send data size to send, and get to-receive sizes
  std::vector<int> send_sizes(outdegree, 0), recv_sizes(indegree);
  std::adjacent_difference(fwd_sharing_offsets.begin() + 1,
                           fwd_sharing_offsets.end(), send_sizes.begin());
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                        MPI_INT, _comm_owner_to_ghost.comm());

  // Work out recv offsets and send/receive
  std::vector<int> recv_offsets(recv_sizes.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   recv_offsets.begin() + 1);
  std::vector<std::int64_t> recv_data(recv_offsets.back());

  // Start data exchange
  MPI_Request request;
  MPI_Ineighbor_alltoallv(
      fwd_sharing_data.data(), send_sizes.data(), fwd_sharing_offsets.data(),
      MPI_INT64_T, recv_data.data(), recv_sizes.data(), recv_offsets.data(),
      MPI_INT64_T, _comm_owner_to_ghost.comm(), &request);

  // For my ghosts, add owning rank to list of sharing ranks
  const std::int32_t size_local = this->size_local();
  const std::vector<int> ghost_owners = this->ghost_owner_rank();
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    shared_indices[size_local + i].insert(ghost_owners[i]);

  // Build map from global index to local index for ghosts
  std::vector<std::pair<std::int64_t, std::int32_t>> ghosts;
  ghosts.reserve(_ghosts.size());
  for (auto idx : _ghosts)
  {
    ghosts.push_back(
        {idx, static_cast<std::int32_t>(size_local + ghosts.size())});
  }
  std::sort(ghosts.begin(), ghosts.end());

  // Wait for all-to-all to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  // Add other ranks that also 'ghost' my ghost indices
  int myrank = -1;
  MPI_Comm_rank(_comm_owner_to_ghost.comm(), &myrank);
  for (std::size_t i = 0; i < recv_data.size();)
  {
    auto it = std::lower_bound(
        ghosts.begin(), ghosts.end(),
        typename decltype(ghosts)::value_type(recv_data[i], 0),
        [](auto& a, auto& b) { return a.first < b.first; });
    assert(it != ghosts.end() and it->first == recv_data[i]);

    const std::int32_t idx = it->second;
    const int set_size = recv_data[i + 1];
    for (int j = 0; j < set_size; ++j)
    {
      if (recv_data[i + 2 + j] != myrank)
        shared_indices[idx].insert(recv_data[i + 2 + +j]);
    }
    i += set_size + 2;
  }

  return shared_indices;
}
//-----------------------------------------------------------------------------
std::pair<IndexMap, std::vector<std::int32_t>>
IndexMap::create_submap(const xtl::span<const std::int32_t>& indices) const
{
  if (!indices.empty() and indices.back() >= this->size_local())
  {
    throw std::runtime_error(
        "Unowned index detected when creating sub-IndexMap");
  }

  MPI_Comm comm = this->comm(Direction::forward);

  int myrank = 0;
  MPI_Comm_rank(comm, &myrank);

  // --- Step 1: Compute new offest for this rank and new global size

  std::int64_t local_size = indices.size();
  std::int64_t offset = 0;
  MPI_Request request_offset;
  MPI_Iexscan(&local_size, &offset, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_offset);

  std::int64_t size_global = 0;
  MPI_Request request_size;
  MPI_Iallreduce(&local_size, &size_global, 1, MPI_INT64_T, MPI_SUM, comm,
                 &request_size);

  // --- Step 2: Create array from old to new index for owned indices,
  // setting entries to -1 if they are not in the new map

  std::vector<std::int32_t> old_to_new_index(this->size_local(), -1);
  for (std::size_t i = 0; i < indices.size(); ++i)
    old_to_new_index[indices[i]] = i;

  // --- Step 3: Compute the destination that the new index map will
  // send to (fwd scatter) and build the new shared_indices adjacency
  // list

  // Loop over ranks that ghost data in the original map
  std::vector<int> ranks_old_to_new_send(_shared_indices->num_nodes(), -1);
  std::vector<std::int32_t> shared_indices_data, shared_indices_off(1, 0);
  for (std::int32_t r_old = 0; r_old < _shared_indices->num_nodes(); ++r_old)
  {
    // For indices sent to old rank r_old, add the new index
    int num_indices = 0;
    for (std::int32_t idx_old : _shared_indices->links(r_old))
    {
      if (auto idx = old_to_new_index[idx_old]; idx >= 0)
      {
        shared_indices_data.push_back(idx);
        ++num_indices;
      }
    }

    // If indices in the new map will be sent to r_old, update
    // old-to-new map and update offset array
    if (num_indices > 0)
    {
      ranks_old_to_new_send[r_old] = shared_indices_off.size() - 1;
      shared_indices_off.push_back(shared_indices_off.back() + num_indices);
    }
  }

  // Create new shared_indices adjacency list
  shared_indices_data.shrink_to_fit();
  shared_indices_off.shrink_to_fit();
  auto shared_indices = std::make_unique<graph::AdjacencyList<std::int32_t>>(
      std::move(shared_indices_data), std::move(shared_indices_off));

  // --- Step 4: Create sizes_send_fwd array

  MPI_Wait(&request_offset, MPI_STATUS_IGNORE);

  // TODO: Can we avoid this step and pack the buffer directly?
  // Build array of new global indices for indices in the new map
  std::vector<std::int64_t> global_indices_new(this->size_local(), -1);
  for (std::size_t i = 0; i < indices.size(); ++i)
    global_indices_new[indices[i]] = i + offset;

  // --- Step 5: Send new global indices to ranks that ghost indices on
  // this rank

  const std::vector<std::int32_t>& send_indices
      = this->scatter_fwd_indices().array();
  std::vector<std::int64_t> buffer_send_fwd(send_indices.size());
  std::transform(
      send_indices.cbegin(), send_indices.cend(), buffer_send_fwd.begin(),
      [&global_indices_new](auto idx) { return global_indices_new[idx]; });

  MPI_Request request1;
  std::vector<std::int64_t> buffer_recv_fwd(this->num_ghosts());
  MPI_Datatype data_type = MPI_INT64_T;
  this->scatter_fwd_begin(xtl::span<const std::int64_t>(buffer_send_fwd),
                          data_type, request1,
                          xtl::span<std::int64_t>(buffer_recv_fwd));
  this->scatter_fwd_end(request1);

  // --- Step 6: Determine which ranks that send data to this rank for
  // *this will also send data to the new map on this rank

  // Count number of ghost from each rank
  std::vector<std::int32_t> ranks_old_to_new_recv(_sizes_recv_fwd.size(), -1),
      displs_recv_fwd(1, 0), sizes_recv_fwd_new(_sizes_recv_fwd.size(), 0);
  for (std::size_t r_old = 0; r_old < _sizes_recv_fwd.size(); ++r_old)
  {
    // Count number of ghosts owned by rank r_old
    assert(r_old + 1 < _displs_recv_fwd.size());
    sizes_recv_fwd_new[r_old] = std::count_if(
        std::next(buffer_recv_fwd.cbegin(), _displs_recv_fwd[r_old]),
        std::next(buffer_recv_fwd.cbegin(), _displs_recv_fwd[r_old + 1]),
        [](auto x) { return x >= 0; });

    if (sizes_recv_fwd_new[r_old] > 0)
    {
      // Will receive data from r_old
      ranks_old_to_new_recv[r_old] = displs_recv_fwd.size() - 1;
      displs_recv_fwd.push_back(displs_recv_fwd.back()
                                + sizes_recv_fwd_new[r_old]);
    }
  }

  // --- Step 7: Build ghost_pos_recv_fwd for the new map

  // Build list of ghosts in the new map, and compute the new owning rank
  std::vector<std::int64_t> ghosts;
  std::vector<std::int32_t> ghost_owner, new_to_old_ghost;
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
  {
    // Get new index for old _ghosts[i]
    std::size_t buffer_pos = _ghost_pos_recv_fwd[i];
    if (auto index = buffer_recv_fwd[buffer_pos]; index >= 0)
    {
      // If index is in the new map, add index and compute owning rank
      // (rank index on new map)
      ghosts.push_back(index);
      new_to_old_ghost.push_back(i);
      auto it = std::upper_bound(_displs_recv_fwd.cbegin(),
                                 _displs_recv_fwd.cend(), buffer_pos);
      const int r_old = std::distance(_displs_recv_fwd.cbegin(), it) - 1;
      ghost_owner.push_back(ranks_old_to_new_recv[r_old]);
    }
  }

  // Compute ghost_pos_recv_fwd
  std::vector<std::int32_t> ghost_pos_recv_fwd(ghost_owner.size());
  std::transform(ghost_owner.cbegin(), ghost_owner.cend(),
                 ghost_pos_recv_fwd.begin(),
                 [displs = displs_recv_fwd](auto owner) mutable
                 { return displs[owner]++; });

  // Step 8: Create neighbourhood communicators for the new map

  // Get global src/dest from forward comm (owner-to-ghost) from
  // original map
  const auto [src_ranks, dest_ranks] = dolfinx::MPI::neighbors(comm);
  std::vector<int> in_ranks, out_ranks;
  for (std::size_t r = 0; r < ranks_old_to_new_recv.size(); ++r)
    if (ranks_old_to_new_recv[r] >= 0)
      in_ranks.push_back(src_ranks[r]);
  for (std::size_t r = 0; r < ranks_old_to_new_send.size(); ++r)
    if (ranks_old_to_new_send[r] >= 0)
      out_ranks.push_back(dest_ranks[r]);

  // NOTE: create uniform weights as a workaround to issue
  // https://github.com/pmodels/mpich/issues/5764
  std::vector<int> in_weights(in_ranks.size(), 1);
  std::vector<int> out_weights(out_ranks.size(), 1);

  MPI_Comm comm0, comm1;
  MPI_Dist_graph_create_adjacent(comm, in_ranks.size(), in_ranks.data(),
                                 in_weights.data(), out_ranks.size(),
                                 out_ranks.data(), out_weights.data(),
                                 MPI_INFO_NULL, false, &comm0);
  MPI_Dist_graph_create_adjacent(comm, out_ranks.size(), out_ranks.data(),
                                 out_weights.data(), in_ranks.size(),
                                 in_ranks.data(), in_weights.data(),
                                 MPI_INFO_NULL, false, &comm1);

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request_size, MPI_STATUS_IGNORE);

  displs_recv_fwd.shrink_to_fit();
  ghosts.shrink_to_fit();
  new_to_old_ghost.shrink_to_fit();

  return {IndexMap({offset, offset + local_size}, size_global, _comm.comm(),
                   dolfinx::MPI::Comm(comm0, false),
                   dolfinx::MPI::Comm(comm1, false), std::move(displs_recv_fwd),
                   std::move(ghost_pos_recv_fwd), std::move(ghosts),
                   std::move(shared_indices)),
          std::move(new_to_old_ghost)};
}
//-----------------------------------------------------------------------------
