// Copyright (C) 2015-2022 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include "sort.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::common;

namespace
{

/// @brief Given source ranks (ranks that own indices ghosted by the
/// calling rank), compute ranks that ghost indices owned by the calling
/// rank.
/// @param comm MPI communicator.
/// @param owners List of ranks that own each ghost index.
/// @return (src ranks, destination ranks). Both lists are sorted.
std::array<std::vector<int>, 2> build_src_dest(MPI_Comm comm,
                                               std::span<const int> owners)
{
  if (dolfinx::MPI::size(comm) == 1)
  {
    assert(owners.empty());
    return std::array<std::vector<int>, 2>();
  }

  std::vector<int> src(owners.begin(), owners.end());
  std::sort(src.begin(), src.end());
  src.erase(std::unique(src.begin(), src.end()), src.end());
  src.shrink_to_fit();
  std::vector<int> dest = dolfinx::MPI::compute_graph_edges_nbx(comm, src);
  std::sort(dest.begin(), dest.end());
  return {std::move(src), std::move(dest)};
}

/// @brief Helper function that sends ghost indices on a given process
/// to their owning rank, and receives indices owned by this process
/// that are ghosts on other processes.
///
/// It also returns the data structures used in this common
/// communication pattern.
///
/// @param[in] comm The communicator (global).
/// @param[in] src Source ranks on `comm`.
/// @param[in] dest Destination ranks on `comm`.
/// @param[in] ghosts Ghost indices on calling process.
/// @param[in] owners Owning rank for each entry in `ghosts`.
/// @param[in] include_ghost A list of the same length as `ghosts`, whose
/// ith entry must be non-zero (true) to include `ghost[i]`, otherwise
/// the ghost will be excluded
/// @return 1) The ghost indices packed in a buffer for communication
///         2) The received indices (in receive buffer layout)
///         3) A map relating the position of a ghost in the packed
///            data (1) to to its position in `ghosts`.
///         4) The number of indices to send to each process.
///         5) The number of indices received by each process.
///         6) The send displacements.
///         7) The received displacements.
/// @pre `src` must be sorted and unique
/// @pre `dest` must be sorted and unique
std::tuple<std::vector<std::int64_t>, std::vector<std::int64_t>,
           std::vector<std::size_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>, std::vector<int>, std::vector<int>>
communicate_ghosts_to_owners(MPI_Comm comm, std::span<const int> src,
                             std::span<const int> dest,
                             std::span<const std::int64_t> ghosts,
                             std::span<const std::int32_t> owners,
                             std::span<const std::uint8_t> include_ghost)
{
  // Send ghost indices to owning rank
  std::vector<std::int64_t> send_indices, recv_indices;
  std::vector<std::size_t> ghost_buffer_pos;
  std::vector<std::int32_t> send_sizes, recv_sizes;
  std::vector<int> send_disp, recv_disp;
  {
    // Create neighbourhood comm (ghost -> owner)
    MPI_Comm comm0;
    int ierr = MPI_Dist_graph_create_adjacent(
        comm, dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(), src.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    dolfinx::MPI::check_error(comm, ierr);

    // Pack ghosts indices
    std::vector<std::vector<std::int64_t>> send_data(src.size());
    std::vector<std::vector<std::size_t>> pos_to_ghost(src.size());
    for (std::size_t i = 0; i < ghosts.size(); ++i)
    {
      auto it = std::lower_bound(src.begin(), src.end(), owners[i]);
      assert(it != src.end() and *it == owners[i]);
      std::size_t r = std::distance(src.begin(), it);
      if (include_ghost[i])
      {
        send_data[r].push_back(ghosts[i]);
        pos_to_ghost[r].push_back(i);
      }
      else
      {
        send_data[r].push_back(-1);
        pos_to_ghost[r].push_back(-1);
      }
    }

    // Count number of ghosts per dest
    std::transform(send_data.begin(), send_data.end(),
                   std::back_inserter(send_sizes),
                   [](auto& d) { return d.size(); });

    // Send how many indices I ghost to each owner, and receive how many
    // of my indices other ranks ghost
    recv_sizes.resize(dest.size(), 0);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Request sizes_request;
    MPI_Ineighbor_alltoall(send_sizes.data(), 1, MPI_INT32_T, recv_sizes.data(),
                           1, MPI_INT32_T, comm0, &sizes_request);

    // Build send buffer and ghost position to send buffer position
    for (auto& d : send_data)
      send_indices.insert(send_indices.end(), d.begin(), d.end());
    for (auto& p : pos_to_ghost)
      ghost_buffer_pos.insert(ghost_buffer_pos.end(), p.begin(), p.end());

    // Prepare displacement vectors
    send_disp.resize(src.size() + 1, 0);
    recv_disp.resize(dest.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::next(send_disp.begin()));
    MPI_Wait(&sizes_request, MPI_STATUS_IGNORE);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::next(recv_disp.begin()));

    // Send ghost indices to owner, and receive indices
    recv_indices.resize(recv_disp.back());
    ierr = MPI_Neighbor_alltoallv(send_indices.data(), send_sizes.data(),
                                  send_disp.data(), MPI_INT64_T,
                                  recv_indices.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT64_T, comm0);
    dolfinx::MPI::check_error(comm, ierr);

    ierr = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(comm, ierr);
  }

  return {std::move(send_indices),     std::move(recv_indices),
          std::move(ghost_buffer_pos), std::move(send_sizes),
          std::move(recv_sizes),       std::move(send_disp),
          std::move(recv_disp)};
}

/// Given an index map and a subset of local indices (can be owned or
/// ghost but must be unique and sorted), compute the owned, ghost and
/// ghost owners in the submap.
/// @param[in] imap An index map.
/// @param[in] indices List of entity indices (indices local to the
/// process).
/// @param[in] order Control the order in which ghost indices appear in
/// the new map.
/// @param[in] allow_owner_change Allows indices that are not included
/// by their owning process but included on sharing processes to be
/// included in the submap. These indices will be owned by one of the
/// sharing processes in the submap.
/// @pre `indices` must be sorted and unique.
/// @return The (1) owned, (2) ghost and (3) ghost owners in the submap,
/// and (4) submap src ranks and (5) submap destination ranks. All
/// indices are local and with respect to the original index map.
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<int>, std::vector<int>, std::vector<int>>
compute_submap_indices(const IndexMap& imap,
                       std::span<const std::int32_t> indices,
                       IndexMapOrder order, bool allow_owner_change)
{
  std::span<const int> src = imap.src();
  std::span<const int> dest = imap.dest();

  // Create lookup array to determine if an index is in the sub-map
  std::vector<std::uint8_t> is_in_submap(imap.size_local() + imap.num_ghosts(),
                                         0);
  std::ranges::for_each(indices,
                        [&is_in_submap](auto i) { is_in_submap[i] = 1; });

  // --- Step 1 ---: Send ghost indices in `indices` to their owners
  // and receive indices owned by this process that are in `indices`
  // on other processes
  const auto [send_indices, recv_indices, ghost_buffer_pos, send_sizes,
              recv_sizes, send_disp, recv_disp]
      = communicate_ghosts_to_owners(
          imap.comm(), imap.src(), imap.dest(), imap.ghosts(), imap.owners(),
          std::span(is_in_submap.cbegin() + imap.size_local(),
                    is_in_submap.cend()));

  // --- Step 2 ---: Create a map from the indices in `recv_indices` (i.e.
  // indices owned by this process that are in `indices` on other processes) to
  // their owner in the submap. This is required since not all indices in
  // `recv_indices` will necessarily be in `indices` on this process, and thus
  // other processes must own them in the submap.

  std::vector<int> recv_owners(send_disp.back());
  const int rank = dolfinx::MPI::rank(imap.comm());
  {
    // Flag to track if the owner of any indices have changed in the submap
    bool owners_changed = false;

    // Create a map from (global) indices in `recv_indices` to a list of
    // processes that can own them in the submap.
    std::vector<std::pair<std::int64_t, int>> global_idx_to_possible_owner;
    const std::array local_range = imap.local_range();
    // Loop through the received indices
    for (std::size_t i = 0; i < recv_disp.size() - 1; ++i)
    {
      for (int j = recv_disp[i]; j < recv_disp[i + 1]; ++j)
      {
        // Check if the index is included in the submap by process dest[i].
        // If so, it must be assigned a submap owner
        if (std::int64_t idx = recv_indices[j]; idx != -1)
        {
          // Compute the local index
          std::int32_t idx_local = idx - local_range[0];
          assert(idx_local >= 0);
          assert(idx_local < local_range[1]);

          // Check if index is included in the submap on this process. If so,
          // this process remains its owner in the submap. Otherwise,
          // add the process that sent it to the list of possible owners.
          if (is_in_submap[idx_local])
            global_idx_to_possible_owner.push_back({idx, rank});
          else
          {
            owners_changed = true;
            global_idx_to_possible_owner.push_back({idx, dest[i]});
          }
        }
      }

      if (owners_changed and !allow_owner_change)
        throw std::runtime_error("Index owner change detected!");
    }

    std::sort(global_idx_to_possible_owner.begin(),
              global_idx_to_possible_owner.end());

    // Choose the submap owner for each index in `recv_indices`
    std::vector<int> send_owners;
    send_owners.reserve(recv_indices.size());
    for (auto idx : recv_indices)
    {
      // Check if the index is in the submap. If so, choose its owner in the
      // submap, otherwise send -1
      if (idx != -1)
      {
        // NOTE: Could choose new owner in a way that is is better for
        // load balancing, though the impact is probably only very small
        auto it = std::lower_bound(global_idx_to_possible_owner.begin(),
                                   global_idx_to_possible_owner.end(), idx,
                                   [](auto a, auto b) { return a.first < b; });
        assert(it != global_idx_to_possible_owner.end() and it->first == idx);
        send_owners.push_back(it->second);
      }
      else
        send_owners.push_back(-1);
    }

    // Create neighbourhood comm (owner -> ghost)
    MPI_Comm comm1;
    int ierr = MPI_Dist_graph_create_adjacent(
        imap.comm(), src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
        dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);
    dolfinx::MPI::check_error(imap.comm(), ierr);

    // Send the data
    ierr = MPI_Neighbor_alltoallv(send_owners.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT, recv_owners.data(),
                                  send_sizes.data(), send_disp.data(), MPI_INT,
                                  comm1);
    dolfinx::MPI::check_error(imap.comm(), ierr);

    // Free the communicator
    ierr = MPI_Comm_free(&comm1);
    dolfinx::MPI::check_error(imap.comm(), ierr);
  }

  // --- Step 3 --- : Determine the owned indices, ghost indices, and
  // ghost owners in the submap

  // Local indices (w.r.t. original map) owned by this process in the
  // submap
  std::vector<std::int32_t> submap_owned;

  // Local indices (w.r.t. original map) ghosted by this process in the
  // submap
  std::vector<std::int32_t> submap_ghost;

  // The owners of the submap ghost indices (process
  // submap_ghost_owners[i] owns index submap_ghost[i])
  std::vector<int> submap_ghost_owners;

  {
    // Add owned indices to submap_owned
    for (std::int32_t v : indices)
      if (v < imap.size_local())
        submap_owned.push_back(v);

    // FIXME: Could just create when making send_indices
    std::vector<std::int32_t> send_indices_local(send_indices.size());
    imap.global_to_local(send_indices, send_indices_local);

    // Loop over ghost indices (in the original map) and add to
    // submap_owned if the owning process has decided this process to be
    // the submap owner. Else, add the index and its (possibly new)
    // owner to submap_ghost and submap_ghost_owners respectively.
    for (std::size_t i = 0; i < send_indices.size(); ++i)
    {
      // Check if index is in the submap
      if (std::int64_t global_idx = send_indices[i]; global_idx >= 0)
      {
        std::int32_t local_idx = send_indices_local[i];
        if (int owner = recv_owners[i]; owner == rank)
          submap_owned.push_back(local_idx);
        else
        {
          submap_ghost.push_back(local_idx);
          submap_ghost_owners.push_back(owner);
        }
      }
    }
    std::sort(submap_owned.begin(), submap_owned.end());
  }

  // Get submap source ranks
  std::vector<int> submap_src(submap_ghost_owners.begin(),
                              submap_ghost_owners.end());
  std::sort(submap_src.begin(), submap_src.end());
  submap_src.erase(std::unique(submap_src.begin(), submap_src.end()),
                   submap_src.end());
  submap_src.shrink_to_fit();

  // If required, preserve the order of the ghost indices
  if (order == IndexMapOrder::preserve)
  {
    // Build (old postion, new position) list for ghosts and sort
    std::vector<std::pair<std::int32_t, std::int32_t>> pos;
    pos.reserve(submap_ghost.size());
    for (std::int32_t idx : submap_ghost)
      pos.emplace_back(idx, pos.size());
    std::sort(pos.begin(), pos.end());

    // Order ghosts in the sub-map by their position in the parent map
    std::vector<int> submap_ghost_owners1;
    submap_ghost_owners1.reserve(submap_ghost_owners.size());
    std::vector<std::int32_t> submap_ghost1;
    submap_ghost1.reserve(submap_ghost.size());
    for (auto [_, idx] : pos)
    {
      submap_ghost_owners1.push_back(submap_ghost_owners[idx]);
      submap_ghost1.push_back(submap_ghost[idx]);
    }

    submap_ghost_owners = std::move(submap_ghost_owners1);
    submap_ghost = std::move(submap_ghost1);
  }

  // Compute submap destination ranks
  // FIXME Remove call to NBX
  std::vector<int> submap_dest
      = dolfinx::MPI::compute_graph_edges_nbx(imap.comm(), submap_src);
  std::sort(submap_dest.begin(), submap_dest.end());

  return {std::move(submap_owned), std::move(submap_ghost),
          std::move(submap_ghost_owners), std::move(submap_src),
          std::move(submap_dest)};
}

/// Compute the global indices of ghosts in a submap.
/// @param[in] submap_src The submap source ranks
/// @param[in] submap_dest The submap destination ranks
/// @param[in] submap_owned Owned submap indices (local w.r.t. original
/// index map)
/// @param[in] submap_ghosts_global Ghost submap indices (global w.r.t.
/// original index map)
/// @param[in] submap_ghost_owners The ranks that own the ghosts in the
/// submap
/// @param[in] submap_offset The global offset for this rank in the
/// submap
/// @param[in] imap The original index map
/// @pre submap_owned must be sorted and contain no repeated indices
std::vector<std::int64_t>
compute_submap_ghost_indices(std::span<const int> submap_src,
                             std::span<const int> submap_dest,
                             std::span<const std::int32_t> submap_owned,
                             std::span<const std::int64_t> submap_ghosts_global,
                             std::span<const std::int32_t> submap_ghost_owners,
                             int submap_offset, const IndexMap& imap)
{
  // --- Step 1 ---: Send global ghost indices (w.r.t. original imap) to
  // owning rank

  auto [send_indices, recv_indices, ghost_perm, send_sizes, recv_sizes,
        send_disp, recv_disp]
      = communicate_ghosts_to_owners(
          imap.comm(), submap_src, submap_dest, submap_ghosts_global,
          submap_ghost_owners,
          std::vector<std::uint8_t>(submap_ghosts_global.size(), 1));

  // --- Step 2 ---: For each received index, compute the submap global
  // index

  std::vector<std::int64_t> send_gidx;
  {
    send_gidx.reserve(recv_indices.size());
    // NOTE: Received indices are owned by this process in the submap, but not
    // necessarily in the original imap, so we must use global_to_local to
    // convert rather than subtracting local_range[0]
    // TODO Convert recv_indices or submap_owned?
    std::vector<int32_t> recv_indices_local(recv_indices.size());
    imap.global_to_local(recv_indices, recv_indices_local);

    // Compute submap global index
    for (auto idx : recv_indices_local)
    {
      // Could avoid search by creating look-up array
      auto it = std::lower_bound(submap_owned.begin(), submap_owned.end(), idx);
      assert(it != submap_owned.end() and *it == idx);
      std::size_t idx_local_submap = std::distance(submap_owned.begin(), it);
      send_gidx.push_back(idx_local_submap + submap_offset);
    }
  }

  // --- Step 3 ---: Send submap global indices to process that ghost them

  std::vector<std::int64_t> recv_gidx(send_disp.back());
  {
    // Create neighbourhood comm (owner -> ghost)
    MPI_Comm comm1;
    int ierr = MPI_Dist_graph_create_adjacent(
        imap.comm(), submap_src.size(), submap_src.data(), MPI_UNWEIGHTED,
        submap_dest.size(), submap_dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
        false, &comm1);
    dolfinx::MPI::check_error(imap.comm(), ierr);

    // Send indices to ghosting ranks
    ierr = MPI_Neighbor_alltoallv(send_gidx.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT64_T,
                                  recv_gidx.data(), send_sizes.data(),
                                  send_disp.data(), MPI_INT64_T, comm1);
    dolfinx::MPI::check_error(imap.comm(), ierr);

    ierr = MPI_Comm_free(&comm1);
    dolfinx::MPI::check_error(imap.comm(), ierr);
  }

  // --- Step 4---: Unpack received data

  std::vector<std::int64_t> ghost_submap_gidx(submap_ghosts_global.size());
  for (std::size_t i = 0; i < recv_gidx.size(); ++i)
    ghost_submap_gidx[ghost_perm[i]] = recv_gidx[i];

  return ghost_submap_gidx;
}
} // namespace

//-----------------------------------------------------------------------------
std::vector<int32_t>
common::compute_owned_indices(std::span<const std::int32_t> indices,
                              const IndexMap& map)
{
  // Assume that indices are sorted and unique
  assert(std::is_sorted(indices.begin(), indices.end()));

  std::span ghosts = map.ghosts();
  std::vector<int> owners(map.owners().begin(), map.owners().end());

  // Find first index that is not owned by this rank
  std::int32_t size_local = map.size_local();
  const auto it_owned_end
      = std::lower_bound(indices.begin(), indices.end(), size_local);

  // Get global indices and owners for ghost indices
  std::size_t first_ghost_index = std::distance(indices.begin(), it_owned_end);
  std::int32_t num_ghost_indices = indices.size() - first_ghost_index;
  std::vector<std::int64_t> global_indices(num_ghost_indices);
  std::vector<int> ghost_owners(num_ghost_indices);
  for (std::int32_t i = 0; i < num_ghost_indices; ++i)
  {
    std::int32_t idx = indices[first_ghost_index + i];
    std::int32_t pos = idx - size_local;
    global_indices[i] = ghosts[pos];
    ghost_owners[i] = owners[pos];
  }

  // Sort indices and owners
  std::sort(global_indices.begin(), global_indices.end());
  std::sort(ghost_owners.begin(), ghost_owners.end());

  std::span dest = map.dest();
  std::span src = map.src();

  // Count number of ghost per destination
  std::vector<int> send_sizes(src.size(), 0);
  std::vector<int> send_disp(src.size() + 1, 0);
  auto it = ghost_owners.begin();
  for (std::size_t i = 0; i < src.size(); i++)
  {
    int owner = src[i];
    auto begin = std::find(it, ghost_owners.end(), owner);
    auto end = std::upper_bound(begin, ghost_owners.end(), owner);

    // Count number of ghosts (if any)
    send_sizes[i] = std::distance(begin, end);
    send_disp[i + 1] = send_disp[i] + send_sizes[i];

    if (begin != end)
      it = end;
  }

  // Create ghost -> owner comm
  MPI_Comm comm;
  int ierr = MPI_Dist_graph_create_adjacent(
      map.comm(), dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(),
      src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
  dolfinx::MPI::check_error(map.comm(), ierr);

  // Exchange number of indices to send/receive from each rank
  std::vector<int> recv_sizes(dest.size(), 0);
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(),
                               1, MPI_INT, comm);
  dolfinx::MPI::check_error(comm, ierr);

  // Prepare receive displacement array
  std::vector<int> recv_disp(dest.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_disp.begin()));

  // Send ghost indices to owner, and receive owned indices
  std::vector<std::int64_t> recv_buffer(recv_disp.back());
  std::vector<std::int64_t>& send_buffer = global_indices;
  ierr = MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                                send_disp.data(), MPI_INT64_T,
                                recv_buffer.data(), recv_sizes.data(),
                                recv_disp.data(), MPI_INT64_T, comm);
  dolfinx::MPI::check_error(comm, ierr);
  ierr = MPI_Comm_free(&comm);
  dolfinx::MPI::check_error(map.comm(), ierr);

  // Remove duplicates from received indices
  std::sort(recv_buffer.begin(), recv_buffer.end());
  recv_buffer.erase(std::unique(recv_buffer.begin(), recv_buffer.end()),
                    recv_buffer.end());

  // Copy owned and ghost indices into return array
  std::vector<std::int32_t> owned;
  owned.reserve(num_ghost_indices + recv_buffer.size());
  std::copy(indices.begin(), it_owned_end, std::back_inserter(owned));
  std::transform(recv_buffer.begin(), recv_buffer.end(),
                 std::back_inserter(owned),
                 [range = map.local_range()](auto idx)
                 {
                   assert(idx >= range[0]);
                   assert(idx < range[1]);
                   return idx - range[0];
                 });

  std::sort(owned.begin(), owned.end());
  owned.erase(std::unique(owned.begin(), owned.end()), owned.end());

  return owned;
}
//-----------------------------------------------------------------------------
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
common::stack_index_maps(
    const std::vector<std::pair<std::reference_wrapper<const IndexMap>, int>>&
        maps)
{
  // Compute process offset for stacked index map
  const std::int64_t process_offset = std::accumulate(
      maps.begin(), maps.end(), std::int64_t(0),
      [](std::int64_t c, auto& map) -> std::int64_t
      { return c + map.first.get().local_range()[0] * map.second; });

  // Get local offset (into new map) for each index map
  std::vector<std::int32_t> local_sizes;
  std::transform(maps.begin(), maps.end(), std::back_inserter(local_sizes),
                 [](auto map)
                 { return map.second * map.first.get().size_local(); });
  std::vector<std::int32_t> local_offset(local_sizes.size() + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   std::next(local_offset.begin()));

  // Build list of src ranks (ranks that own ghosts)
  std::set<int> src_set;
  std::set<int> dest_set;
  for (auto& [map, _] : maps)
  {
    std::span _src = map.get().src();
    std::span _dest = map.get().dest();
    src_set.insert(_src.begin(), _src.end());
    dest_set.insert(_dest.begin(), _dest.end());
  }

  std::vector<int> src(src_set.begin(), src_set.end());
  std::vector<int> dest(dest_set.begin(), dest_set.end());

  // Create neighbour comms (0: ghost -> owner, 1: (owner -> ghost)
  MPI_Comm comm0, comm1;
  int ierr = MPI_Dist_graph_create_adjacent(
      maps.at(0).first.get().comm(), dest.size(), dest.data(), MPI_UNWEIGHTED,
      src.size(), src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
  dolfinx::MPI::check_error(maps.at(0).first.get().comm(), ierr);
  ierr = MPI_Dist_graph_create_adjacent(
      maps.at(0).first.get().comm(), src.size(), src.data(), MPI_UNWEIGHTED,
      dest.size(), dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);
  dolfinx::MPI::check_error(maps.at(0).first.get().comm(), ierr);

  // NOTE: We could perform each MPI call just once rather than per map,
  // but the complexity may not be worthwhile since this function is
  // typically used for 'block' (rather the nested) problems, which is
  // not the most efficient approach anyway.

  std::vector<std::vector<std::int64_t>> ghosts_new(maps.size());
  std::vector<std::vector<int>> ghost_owners_new(maps.size());

  // For each map, send ghost indices to owner and owners send back the
  // new index
  for (std::size_t m = 0; m < maps.size(); ++m)
  {
    const int bs = maps[m].second;
    const IndexMap& map = maps[m].first.get();
    std::span ghosts = map.ghosts();
    std::span owners = map.owners();

    // For each owning rank (on comm), create vector of this rank's
    // ghosts
    std::vector<std::int64_t> send_indices;
    std::vector<std::int32_t> send_sizes;
    std::vector<std::size_t> ghost_buffer_pos;
    {
      std::vector<std::vector<std::int64_t>> ghost_by_rank(src.size());
      std::vector<std::vector<std::size_t>> pos_to_ghost(src.size());
      for (std::size_t i = 0; i < ghosts.size(); ++i)
      {
        auto it = std::lower_bound(src.begin(), src.end(), owners[i]);
        assert(it != src.end() and *it == owners[i]);
        int r = std::distance(src.begin(), it);
        ghost_by_rank[r].push_back(ghosts[i]);
        pos_to_ghost[r].push_back(i);
      }

      // Count number of ghosts per dest
      std::transform(ghost_by_rank.begin(), ghost_by_rank.end(),
                     std::back_inserter(send_sizes),
                     [](auto& g) { return g.size(); });

      // Send buffer and ghost position to send buffer position
      for (auto& g : ghost_by_rank)
        send_indices.insert(send_indices.end(), g.begin(), g.end());
      for (auto& p : pos_to_ghost)
        ghost_buffer_pos.insert(ghost_buffer_pos.end(), p.begin(), p.end());
    }

    // Send how many indices I ghost to each owner, and receive how many
    // of my indices other ranks ghost
    std::vector<std::int32_t> recv_sizes(dest.size(), 0);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT32_T,
                                 recv_sizes.data(), 1, MPI_INT32_T, comm0);
    dolfinx::MPI::check_error(comm0, ierr);

    // Prepare displacement vectors
    std::vector<int> send_disp(src.size() + 1, 0),
        recv_disp(dest.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::next(send_disp.begin()));
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::next(recv_disp.begin()));

    // Send ghost indices to owner, and receive indices
    std::vector<std::int64_t> recv_indices(recv_disp.back());
    ierr = MPI_Neighbor_alltoallv(send_indices.data(), send_sizes.data(),
                                  send_disp.data(), MPI_INT64_T,
                                  recv_indices.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT64_T, comm0);
    dolfinx::MPI::check_error(comm0, ierr);

    // For each received index (which I should own), compute its new
    // index in the concatenated index map
    std::vector<std::int64_t> ghost_old_to_new;
    ghost_old_to_new.reserve(recv_indices.size());
    std::int64_t offset_old = map.local_range()[0];
    std::int64_t offset_new = local_offset[m] + process_offset;
    for (std::int64_t idx : recv_indices)
    {
      auto idx_local = idx - offset_old;
      assert(idx_local >= 0);
      ghost_old_to_new.push_back(bs * idx_local + offset_new);
    }

    // Send back/receive new indices
    std::vector<std::int64_t> ghosts_new_idx(send_disp.back());
    ierr = MPI_Neighbor_alltoallv(ghost_old_to_new.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT64_T,
                                  ghosts_new_idx.data(), send_sizes.data(),
                                  send_disp.data(), MPI_INT64_T, comm1);
    dolfinx::MPI::check_error(comm1, ierr);

    // Unpack new indices and store owner
    std::vector<std::int64_t>& ghost_idx = ghosts_new[m];
    ghost_idx.resize(bs * map.ghosts().size());
    std::vector<int>& owners_new = ghost_owners_new[m];
    owners_new.resize(bs * map.ghosts().size());
    for (std::size_t i = 0; i < send_disp.size() - 1; ++i)
    {
      int rank = src[i];
      for (int j = send_disp[i]; j < send_disp[i + 1]; ++j)
      {
        std::size_t p = ghost_buffer_pos[j];
        for (int k = 0; k < bs; ++k)
        {
          ghost_idx[bs * p + k] = ghosts_new_idx[j] + k;
          owners_new[bs * p + k] = rank;
        }
      }
    }
  }

  // Destroy communicators
  ierr = MPI_Comm_free(&comm0);
  dolfinx::MPI::check_error(maps.at(0).first.get().comm(), ierr);

  ierr = MPI_Comm_free(&comm1);
  dolfinx::MPI::check_error(maps.at(0).first.get().comm(), ierr);

  return {process_offset, std::move(local_offset), std::move(ghosts_new),
          std::move(ghost_owners_new)};
}
//-----------------------------------------------------------------------------
std::pair<IndexMap, std::vector<std::int32_t>>
common::create_sub_index_map(const IndexMap& imap,
                             std::span<const std::int32_t> indices,
                             IndexMapOrder order, bool allow_owner_change)
{
  // Compute the owned, ghost, and ghost owners of submap indices.
  // NOTE: All indices are local and numbered w.r.t. the original (imap)
  // index map
  auto [submap_owned, submap_ghost, submap_ghost_owners, submap_src,
        submap_dest]
      = compute_submap_indices(imap, indices, order, allow_owner_change);

  // Compute submap offset for this rank
  std::int64_t submap_local_size = submap_owned.size();
  std::int64_t submap_offset = 0;
  int ierr = MPI_Exscan(&submap_local_size, &submap_offset, 1, MPI_INT64_T,
                        MPI_SUM, imap.comm());
  dolfinx::MPI::check_error(imap.comm(), ierr);

  // Compute the global indices (w.r.t. the submap) of the submap ghosts
  std::vector<std::int64_t> submap_ghost_global(submap_ghost.size());
  imap.local_to_global(submap_ghost, submap_ghost_global);
  std::vector<std::int64_t> submap_ghost_gidxs = compute_submap_ghost_indices(
      submap_src, submap_dest, submap_owned, submap_ghost_global,
      submap_ghost_owners, submap_offset, imap);

  // Create a map from (local) indices in the submap to the corresponding
  // (local) index in the original map
  std::vector<std::int32_t> sub_imap_to_imap;
  sub_imap_to_imap.reserve(submap_owned.size() + submap_ghost.size());
  sub_imap_to_imap.insert(sub_imap_to_imap.end(), submap_owned.begin(),
                          submap_owned.end());
  sub_imap_to_imap.insert(sub_imap_to_imap.end(), submap_ghost.begin(),
                          submap_ghost.end());

  return {IndexMap(imap.comm(), submap_local_size, {submap_src, submap_dest},
                   submap_ghost_gidxs, submap_ghost_owners),
          std::move(sub_imap_to_imap)};
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm comm, std::int32_t local_size) : _comm(comm, true)
{
  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = local_size;
  MPI_Request request_scan;
  int ierr = MPI_Iexscan(&local_size_tmp, &offset, 1, MPI_INT64_T, MPI_SUM,
                         _comm.comm(), &request_scan);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  // Send local size to sum reduction to get global size
  MPI_Request request;
  ierr = MPI_Iallreduce(&local_size_tmp, &_size_global, 1, MPI_INT64_T, MPI_SUM,
                        comm, &request);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  ierr = MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(_comm.comm(), ierr);
  _local_range = {offset, offset + local_size};

  // Wait for the MPI_Iallreduce to complete
  ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(_comm.comm(), ierr);
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm comm, std::int32_t local_size,
                   std::span<const std::int64_t> ghosts,
                   std::span<const int> owners)
    : IndexMap(comm, local_size, build_src_dest(comm, owners), ghosts, owners)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm comm, std::int32_t local_size,
                   const std::array<std::vector<int>, 2>& src_dest,
                   std::span<const std::int64_t> ghosts,
                   std::span<const int> owners)
    : _comm(comm, true), _ghosts(ghosts.begin(), ghosts.end()),
      _owners(owners.begin(), owners.end()), _src(src_dest[0]),
      _dest(src_dest[1])
{
  assert(ghosts.size() == owners.size());
  assert(std::is_sorted(src_dest[0].begin(), src_dest[0].end()));
  assert(std::is_sorted(src_dest[1].begin(), src_dest[1].end()));

  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = local_size;
  MPI_Request request_scan;
  int ierr = MPI_Iexscan(&local_size_tmp, &offset, 1, MPI_INT64_T, MPI_SUM,
                         comm, &request_scan);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  // Send local size to sum reduction to get global size
  MPI_Request request;
  ierr = MPI_Iallreduce(&local_size_tmp, &_size_global, 1, MPI_INT64_T, MPI_SUM,
                        comm, &request);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  // Wait for MPI_Iexscan to complete (get offset)
  ierr = MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(_comm.comm(), ierr);
  _local_range = {offset, offset + local_size};

  // Wait for the MPI_Iallreduce to complete
  ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(_comm.comm(), ierr);
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
std::span<const std::int64_t> IndexMap::ghosts() const noexcept
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
void IndexMap::local_to_global(std::span<const std::int32_t> local,
                               std::span<std::int64_t> global) const
{
  assert(local.size() <= global.size());
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  std::transform(
      local.begin(), local.end(), global.begin(),
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
void IndexMap::global_to_local(std::span<const std::int64_t> global,
                               std::span<std::int32_t> local) const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local(
      _ghosts.size());
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    global_to_local[i] = {_ghosts[i], i + local_size};

  std::sort(global_to_local.begin(), global_to_local.end());
  std::transform(global.begin(), global.end(), local.begin(),
                 [range = _local_range,
                  &global_to_local](std::int64_t index) -> std::int32_t
                 {
                   if (index >= range[0] and index < range[1])
                     return index - range[0];
                   else
                   {
                     auto it = std::lower_bound(
                         global_to_local.begin(), global_to_local.end(), index,
                         [](auto a, auto b) { return a.first < b; });
                     return (it != global_to_local.end() and it->first == index)
                                ? it->second
                                : -1;
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
MPI_Comm IndexMap::comm() const { return _comm.comm(); }
//----------------------------------------------------------------------------
graph::AdjacencyList<int> IndexMap::index_to_dest_ranks() const
{
  const std::int64_t offset = _local_range[0];

  // Build lists of src and dest ranks
  std::vector<int> src = _owners;
  std::sort(src.begin(), src.end());
  src.erase(std::unique(src.begin(), src.end()), src.end());
  auto dest = dolfinx::MPI::compute_graph_edges_nbx(_comm.comm(), src);
  std::sort(dest.begin(), dest.end());

  // Array (local idx, ghosting rank) pairs for owned indices
  std::vector<std::pair<std::int32_t, int>> idx_to_rank;

  // 1. Build adjacency list data for owned indices (index, [sharing
  //    ranks])
  std::vector<std::int32_t> offsets = {0};
  std::vector<int> data;
  {
    // Build list of (owner rank, index) pairs for each ghost index, and sort
    std::vector<std::pair<int, std::int64_t>> owner_to_ghost;
    std::transform(_ghosts.begin(), _ghosts.end(), _owners.begin(),
                   std::back_inserter(owner_to_ghost),
                   [](auto idx, auto r) -> std::pair<int, std::int64_t>
                   { return {r, idx}; });
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

    // Create ghost -> owner comm
    MPI_Comm comm0;
    int ierr = MPI_Dist_graph_create_adjacent(
        _comm.comm(), dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(),
        src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    dolfinx::MPI::check_error(_comm.comm(), ierr);

    // Exchange number of indices to send/receive from each rank
    std::vector<int> recv_sizes(dest.size(), 0);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT,
                                 recv_sizes.data(), 1, MPI_INT, comm0);
    dolfinx::MPI::check_error(_comm.comm(), ierr);

    // Prepare receive displacement array
    std::vector<int> recv_disp(dest.size() + 1, 0);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::next(recv_disp.begin()));

    // Send ghost indices to owner, and receive owned indices
    std::vector<std::int64_t> recv_buffer(recv_disp.back());
    ierr = MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                                  send_disp.data(), MPI_INT64_T,
                                  recv_buffer.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT64_T, comm0);
    dolfinx::MPI::check_error(_comm.comm(), ierr);
    ierr = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(_comm.comm(), ierr);

    // Build array of (local index, ghosting local rank), and sort
    for (std::size_t r = 0; r < recv_disp.size() - 1; ++r)
      for (int j = recv_disp[r]; j < recv_disp[r + 1]; ++j)
        idx_to_rank.push_back({recv_buffer[j] - offset, r});
    std::sort(idx_to_rank.begin(), idx_to_rank.end());

    // -- Send to ranks that ghost my indices all the sharing ranks

    // Build adjacency list data for (owned index) -> (ghosting ranks)
    data.reserve(idx_to_rank.size());
    std::transform(idx_to_rank.begin(), idx_to_rank.end(),
                   std::back_inserter(data), [](auto x) { return x.second; });
    offsets.reserve(this->size_local() + this->num_ghosts() + 1);
    {
      auto it = idx_to_rank.begin();

      // Loop over owned indices
      for (std::int32_t i = 0; i < this->size_local(); ++i)
      {
        auto it1 = std::find_if(it, idx_to_rank.end(),
                                [i](auto x) { return x.first != i; });
        offsets.push_back(offsets.back() + std::distance(it, it1));
        it = it1;
      }
    }
  }

  // 2. Build and add adjacency list data for non-owned indices
  //    (index, [sharing ranks]). Non-owned indices are ghosted but
  //    not owned by this rank.
  {

    // Send data for owned indices back to ghosting ranks (this is
    // necessary to share with ghosting ranks all the ranks that also
    // ghost a ghost index)
    std::vector<std::int64_t> send_buffer;
    std::vector<int> send_sizes;
    {
      const int rank = dolfinx::MPI::rank(_comm.comm());
      std::vector<std::vector<std::int64_t>> dest_idx_to_rank(dest.size());
      for (std::size_t n = 0; n < offsets.size() - 1; ++n)
      {
        std::span<const std::int32_t> ranks(data.data() + offsets[n],
                                            offsets[n + 1] - offsets[n]);
        for (auto r0 : ranks)
        {
          for (auto r : ranks)
          {
            assert(r0 < (int)dest_idx_to_rank.size());
            if (r0 != r)
            {
              dest_idx_to_rank[r0].push_back(n + offset);
              dest_idx_to_rank[r0].push_back(dest[r]);
            }
          }
          dest_idx_to_rank[r0].push_back(n + offset);
          dest_idx_to_rank[r0].push_back(rank);
        }
      }

      // Count number of ghosts per destination and build send buffer
      std::transform(dest_idx_to_rank.begin(), dest_idx_to_rank.end(),
                     std::back_inserter(send_sizes),
                     [](auto& x) { return x.size(); });
      for (auto& d : dest_idx_to_rank)
        send_buffer.insert(send_buffer.end(), d.begin(), d.end());

      // Create owner -> ghost comm
      MPI_Comm comm;
      int ierr = MPI_Dist_graph_create_adjacent(
          _comm.comm(), src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
          dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
      dolfinx::MPI::check_error(_comm.comm(), ierr);

      // Send how many indices I ghost to each owner, and receive how
      // many of my indices other ranks ghost
      std::vector<int> recv_sizes(src.size(), 0);
      send_sizes.reserve(1);
      recv_sizes.reserve(1);
      ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT,
                                   recv_sizes.data(), 1, MPI_INT, comm);
      dolfinx::MPI::check_error(_comm.comm(), ierr);

      // Prepare displacement vectors
      std::vector<int> send_disp(dest.size() + 1, 0),
          recv_disp(src.size() + 1, 0);
      std::partial_sum(send_sizes.begin(), send_sizes.end(),
                       std::next(send_disp.begin()));
      std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                       std::next(recv_disp.begin()));

      std::vector<std::int64_t> recv_indices(recv_disp.back());
      ierr = MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                                    send_disp.data(), MPI_INT64_T,
                                    recv_indices.data(), recv_sizes.data(),
                                    recv_disp.data(), MPI_INT64_T, comm);
      dolfinx::MPI::check_error(_comm.comm(), ierr);
      ierr = MPI_Comm_free(&comm);
      dolfinx::MPI::check_error(_comm.comm(), ierr);

      // Build list of (ghost index, ghost position) pairs for indices
      // ghosted by this rank, and sort
      std::vector<std::pair<std::int64_t, std::int32_t>> idx_to_pos;
      for (auto idx : _ghosts)
        idx_to_pos.push_back({idx, idx_to_pos.size()});
      std::sort(idx_to_pos.begin(), idx_to_pos.end());

      // Build list of (local ghost position, sharing rank) pairs from
      // the received data, and sort
      std::vector<std::pair<std::int32_t, int>> idxpos_to_rank;
      for (std::size_t i = 0; i < recv_indices.size(); i += 2)
      {
        std::int64_t idx = recv_indices[i];
        auto it = std::lower_bound(
            idx_to_pos.begin(), idx_to_pos.end(),
            std::pair<std::int64_t, std::int32_t>{idx, 0},
            [](auto a, auto b) { return a.first < b.first; });
        assert(it != idx_to_pos.end() and it->first == idx);

        int rank = recv_indices[i + 1];
        idxpos_to_rank.push_back({it->second, rank});
      }
      std::sort(idxpos_to_rank.begin(), idxpos_to_rank.end());

      // Add processed received data to adjacency list data array, and
      // extend offset array
      std::transform(idxpos_to_rank.begin(), idxpos_to_rank.end(),
                     std::back_inserter(data), [](auto x) { return x.second; });
      auto it = idxpos_to_rank.begin();
      for (std::size_t i = 0; i < _ghosts.size(); ++i)
      {
        auto it1
            = std::find_if(it, idxpos_to_rank.end(), [i](auto x)
                           { return x.first != static_cast<std::int32_t>(i); });
        offsets.push_back(offsets.back() + std::distance(it, it1));
        it = it1;
      }
    }
  }

  // Convert ranks for owned indices from neighbour to global ranks
  std::transform(idx_to_rank.begin(), idx_to_rank.end(), data.begin(),
                 [&dest](auto x) { return dest[x.second]; });

  return graph::AdjacencyList(std::move(data), std::move(offsets));
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> IndexMap::shared_indices() const
{
  // Each process gets a chunk of consecutive indices (global indices)
  // Sorting the ghosts groups them by owner
  std::vector<std::int64_t> send_buffer(_ghosts);
  std::sort(send_buffer.begin(), send_buffer.end());

  std::vector<int32_t> owners(_owners);
  std::sort(owners.begin(), owners.end());
  std::vector<int> send_sizes, send_disp{0};

  // Count number of ghost per destination
  auto it = owners.begin();
  while (it != owners.end())
  {
    auto it1 = std::upper_bound(it, owners.end(), *it);
    send_sizes.push_back(std::distance(it, it1));
    send_disp.push_back(send_disp.back() + send_sizes.back());

    // Advance iterator
    it = it1;
  }

  // Create ghost -> owner comm
  MPI_Comm comm;
  int ierr = MPI_Dist_graph_create_adjacent(
      _comm.comm(), _dest.size(), _dest.data(), MPI_UNWEIGHTED, _src.size(),
      _src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  std::vector<int> recv_sizes(_dest.size(), 0);
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(),
                               1, MPI_INT, comm);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  // Prepare receive displacement array
  std::vector<int> recv_disp(_dest.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_disp.begin()));

  // Send ghost indices to owner, and receive owned indices
  std::vector<std::int64_t> recv_buffer(recv_disp.back());
  ierr = MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                                send_disp.data(), MPI_INT64_T,
                                recv_buffer.data(), recv_sizes.data(),
                                recv_disp.data(), MPI_INT64_T, comm);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  ierr = MPI_Comm_free(&comm);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  std::vector<std::int32_t> shared;
  shared.reserve(recv_buffer.size());
  std::transform(recv_buffer.begin(), recv_buffer.end(),
                 std::back_inserter(shared),
                 [range = _local_range](auto idx)
                 {
                   assert(idx >= range[0]);
                   assert(idx < range[1]);
                   return idx - range[0];
                 });

  // Sort and remove duplicates
  std::sort(shared.begin(), shared.end());
  shared.erase(std::unique(shared.begin(), shared.end()), shared.end());

  return shared;
}
//-----------------------------------------------------------------------------
std::span<const int> IndexMap::src() const noexcept { return _src; }
//-----------------------------------------------------------------------------
std::span<const int> IndexMap::dest() const noexcept { return _dest; }
//-----------------------------------------------------------------------------
std::array<double, 2> IndexMap::imbalance() const
{
  std::array<double, 2> imbalance{-1., -1.};
  std::array<std::int32_t, 2> max_count;
  std::array<std::int32_t, 2> local_sizes
      = {static_cast<std::int32_t>(_local_range[1] - _local_range[0]),
         static_cast<std::int32_t>(_ghosts.size())};

  // Find the maximum number of owned indices and the maximum number of ghost
  // indices across all processes.
  MPI_Allreduce(local_sizes.data(), max_count.data(), 2, MPI_INT32_T, MPI_MAX,
                _comm.comm());

  std::int32_t total_num_ghosts = 0;
  MPI_Allreduce(&local_sizes[1], &total_num_ghosts, 1, MPI_INT32_T, MPI_SUM,
                _comm.comm());

  // Compute the average number of owned and ghost indices per process.
  int comm_size = dolfinx::MPI::size(_comm.comm());
  double avg_owned = static_cast<double>(_size_global) / comm_size;
  double avg_ghosts = static_cast<double>(total_num_ghosts) / comm_size;

  // Compute the imbalance by dividing the maximum number of indices by the
  // corresponding average.
  if (avg_owned > 0)
    imbalance[0] = max_count[0] / avg_owned;
  if (avg_ghosts > 0)
    imbalance[1] = max_count[1] / avg_ghosts;

  return imbalance;
}
//-----------------------------------------------------------------------------
