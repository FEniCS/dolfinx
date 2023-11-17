// Copyright (C) 2015-2022 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include "sort.h"
#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::common;

namespace
{
std::array<std::vector<int>, 2> build_src_dest(MPI_Comm comm,
                                               std::span<const int> owners)
{
  std::vector<int> src(owners.begin(), owners.end());
  std::sort(src.begin(), src.end());
  src.erase(std::unique(src.begin(), src.end()), src.end());
  src.shrink_to_fit();

  std::vector<int> dest = dolfinx::MPI::compute_graph_edges_nbx(comm, src);
  std::sort(dest.begin(), dest.end());

  return {std::move(src), std::move(dest)};
}

std::tuple<std::vector<std::int64_t>, std::vector<std::int64_t>,
           std::vector<std::size_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>, std::vector<int>, std::vector<int>>
communicate_ghosts_to_owners(const MPI_Comm comm,
                             std::span<const std::int32_t> src,
                             std::span<const std::int32_t> dest,
                             std::span<const std::int64_t> ghosts,
                             std::span<const std::int32_t> owners,
                             std::span<const int> include_ghost)
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
      int r = std::distance(src.begin(), it);
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

    // Build send buffer and ghost position to send buffer position
    for (auto& d : send_data)
      send_indices.insert(send_indices.end(), d.begin(), d.end());
    for (auto& p : pos_to_ghost)
      ghost_buffer_pos.insert(ghost_buffer_pos.end(), p.begin(), p.end());

    // Send how many indices I ghost to each owner, and receive how many
    // of my indices other ranks ghost
    recv_sizes.resize(dest.size(), 0);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT32_T,
                                 recv_sizes.data(), 1, MPI_INT32_T, comm0);
    dolfinx::MPI::check_error(comm, ierr);

    // Prepare displacement vectors
    send_disp.resize(src.size() + 1, 0);
    recv_disp.resize(dest.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::next(send_disp.begin()));
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

/// Given an index map and a subset of local indices (can be owned or ghost but
/// must be unique and sorted), computes the owned, ghost and ghost owners in
/// the submap.
/// @param[in] imap An index map.
/// @param[in] indices List of entity indices (indices local to the process).
/// @pre `indices` must be sorted and unique.
/// @return The owned, ghost, and ghost owners in the submap. All indices are
/// local and with respect to the original index map.
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>>
compute_submap_indices(const dolfinx::common::IndexMap& imap,
                       std::span<const std::int32_t> indices)
{
  const MPI_Comm comm = imap.comm();
  std::span<const std::int32_t> src = imap.src();
  std::span<const std::int32_t> dest = imap.dest();

  // Lookup array to determine if an index is in the sub-map
  std::vector<int> is_in_submap(imap.size_local() + imap.num_ghosts(), 0);
  std::for_each(indices.begin(), indices.end(),
                [&is_in_submap](int i) { is_in_submap[i] = 1; });

  // --- Step 1 ---: Send ghost indices in `indices` to their owners
  // and receive indices owned by this process that are in `indices`
  // on other processes
  auto [send_indices, recv_indices, ghost_buffer_pos, send_sizes, recv_sizes,
        send_disp, recv_disp]
      = communicate_ghosts_to_owners(
          imap.comm(), imap.src(), imap.dest(), imap.ghosts(), imap.owners(),
          std::span<const int>(is_in_submap.cbegin() + imap.size_local(),
                               is_in_submap.cend()));

  // --- Step 2 ---: Create a map from the indices in `recv_indices` (i.e.
  // indices owned by this process that are in `indices` on other processes) to
  // their owner in the submap. This is required since not all indices in
  // `recv_indices` will necessarily be in `indices` on this process, and thus
  // other processes must own them in the submap.

  std::vector<std::int32_t> recv_owners(send_disp.back());
  const int rank = dolfinx::MPI::rank(comm);
  {
    // Create a map from (global) owned shared indices owned to processes that
    // can own them.
    // FIXME Avoid map
    // A map from (global) indices in `recv_indices` to a list of processes that
    // can own the index in the submap.
    std::map<std::int64_t, std::vector<std::int32_t>>
        global_idx_to_possible_owner;
    const std::array local_range = imap.local_range();
    for (std::size_t i = 0; i < recv_disp.size() - 1; ++i)
    {
      for (int j = recv_disp[i]; j < recv_disp[i + 1]; ++j)
      {
        // Check that the index is in the submap
        if (std::int64_t idx = recv_indices[j]; idx != -1)
        {
          // Compute the local index
          std::int32_t idx_local = idx - local_range[0];
          assert(idx_local >= 0);
          assert(idx_local < local_range[1]);

          // Check if index is in the submap on this process. If so, this
          // process remains its owner in the submap. Otherwise, add the process
          // that sent it to the list of possible owners.
          if (is_in_submap[idx_local])
            global_idx_to_possible_owner[idx].push_back(rank);
          else
            global_idx_to_possible_owner[idx].push_back(dest[i]);
        }
      }
    }

    // Choose the submap owner for each index in `recv_indices`
    std::vector<std::int32_t> send_owners;
    send_owners.reserve(recv_indices.size());
    for (auto idx : recv_indices)
    {
      // Check the index is in the submap, otherwise send -1
      if (idx != -1)
      {
        // TODO Choose new owner randomly for load balancing
        const std::vector<std::int32_t>& possible_owners
            = global_idx_to_possible_owner[idx];
        // const int random_index = std::rand() % possible_owners.size();
        // send_owners.push_back(possible_owners[random_index]);

        // FIXME TEMPORARILY CHOOSE THE FIRST PROCESS, MAKE RANDOM
        send_owners.push_back(possible_owners[0]);
      }
      else
        send_owners.push_back(-1);
    }

    // Create neighbourhood comm (owner -> ghost)
    MPI_Comm comm1;
    int ierr = MPI_Dist_graph_create_adjacent(
        comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);
    dolfinx::MPI::check_error(comm, ierr);

    // Send the data
    ierr = MPI_Neighbor_alltoallv(send_owners.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT32_T,
                                  recv_owners.data(), send_sizes.data(),
                                  send_disp.data(), MPI_INT32_T, comm1);
    dolfinx::MPI::check_error(comm, ierr);

    // Free the communicator
    ierr = MPI_Comm_free(&comm1);
    dolfinx::MPI::check_error(comm, ierr);
  }

  // --- Step 3 --- : Determine the owned indices, ghost indices, and ghost
  // owners in the submap

  // Local indices (w.r.t. original map) owned by this process in the submap
  std::vector<std::int32_t> submap_owned;

  // Local indices (w.r.t. original map) ghosted by this process in the submap
  std::vector<std::int32_t> submap_ghost;

  // The owners of the submap ghost indices (process submap_ghost_owners[i] owns
  // index submap_ghost[i])
  std::vector<std::int32_t> submap_ghost_owners;

  {
    // Add owned indices to submap_owned
    for (std::int32_t v : indices)
      if (v < imap.size_local())
        submap_owned.push_back(v);

    // FIXME Could just create when making send_indices
    std::vector<std::int32_t> send_indices_local(send_indices.size());
    imap.global_to_local(send_indices, send_indices_local);

    // Loop over ghost indices (in the original map) and add to submap_owned
    // if the owning process has decided this process to be the submap owner.
    // Else, add the index and its (possibly new) owner to submap_ghost and
    // submap_ghost_owners respectively.
    for (std::size_t i = 0; i < send_indices.size(); ++i)
    {
      // Check if index is in the submap
      if (std::int64_t global_idx = send_indices[i]; global_idx >= 0)
      {
        std::int32_t local_idx = send_indices_local[i];
        if (std::int32_t owner = recv_owners[i]; owner == rank)
        {
          submap_owned.push_back(local_idx);
        }
        else
        {
          submap_ghost.push_back(local_idx);
          submap_ghost_owners.push_back(owner);
        }
      }
    }
    std::sort(submap_owned.begin(), submap_owned.end());
  }

  return {std::move(submap_owned), std::move(submap_ghost),
          std::move(submap_ghost_owners)};
}

/// Computes the global indices of ghosts in a submap.
/// @param[in] submap_src The submap source ranks
/// @param[in] submap_dest The submap destination ranks
/// @param[in] submap_owned Owned submap indices (local w.r.t. original index
/// map)
/// @param[in] submap_ghosts_global Ghost submap indices (global w.r.t. original
/// index map)
/// @param[in] submap_ghost_owners The ranks that own the ghosts in the submap
/// @param[in] submap_offset The global offset for this rank in the submap
/// @param[in] imap The original index map
/// @pre submap_owned must be sorted and contain no repeated indices
std::vector<std::int64_t> compute_submap_ghost_indices(
    std::span<const int> submap_src, std::span<const int> submap_dest,
    std::span<const std::int32_t> submap_owned,
    std::span<const std::int64_t> submap_ghosts_global,
    std::span<const std::int32_t> submap_ghost_owners, const int submap_offset,
    const dolfinx::common::IndexMap& imap)
{
  // --- Step 1 ---: Send global ghost indices (w.r.t. original imap) to owning
  // rank
  const MPI_Comm comm = imap.comm();

  auto [send_indices, recv_indices, ghost_perm, send_sizes, recv_sizes,
        send_disp, recv_disp]
      = communicate_ghosts_to_owners(
          imap.comm(), submap_src, submap_dest, submap_ghosts_global,
          submap_ghost_owners,
          std::vector<int>(submap_ghosts_global.size(), 1));

  // --- Step 2 ---: For each received index, compute the submap global index

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
        comm, submap_src.size(), submap_src.data(), MPI_UNWEIGHTED,
        submap_dest.size(), submap_dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
        false, &comm1);
    dolfinx::MPI::check_error(comm, ierr);

    // Send indices to ghosting ranks
    ierr = MPI_Neighbor_alltoallv(send_gidx.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT64_T,
                                  recv_gidx.data(), send_sizes.data(),
                                  send_disp.data(), MPI_INT64_T, comm1);
    dolfinx::MPI::check_error(comm, ierr);

    ierr = MPI_Comm_free(&comm1);
    dolfinx::MPI::check_error(comm, ierr);
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

  std::vector<std::int64_t> ghosts = map.ghosts();
  std::vector<int> owners = map.owners();

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

  const std::vector<int>& dest = map.dest();
  const std::vector<int>& src = map.src();

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
  dolfinx::MPI::check_error(comm, ierr);

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
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
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
    const std::vector<int>& _src = map.get().src();
    const std::vector<int>& _dest = map.get().dest();

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
    const common::IndexMap& map = maps[m].first.get();
    const std::vector<std::int64_t>& ghosts = map.ghosts();
    const std::vector<int>& owners = map.owners();

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
common::create_submap_conn(const dolfinx::common::IndexMap& imap,
                           std::span<const std::int32_t> indices)
{
  const MPI_Comm comm = imap.comm();

  // Compute the owned, ghost, and ghost owners of submap indices.
  // NOTE: All indices are local and numbered w.r.t. the original (imap) index
  // map
  auto [submap_owned, submap_ghost, submap_ghost_owners]
      = compute_submap_indices(imap, indices);

  // Compute submap offset for this rank
  std::int64_t submap_local_size = submap_owned.size();
  std::int64_t submap_offset = 0;
  int ierr = MPI_Exscan(&submap_local_size, &submap_offset, 1, MPI_INT64_T,
                        MPI_SUM, comm);
  dolfinx::MPI::check_error(comm, ierr);

  // Get submap source ranks
  std::vector<int> submap_src(submap_ghost_owners.begin(),
                              submap_ghost_owners.end());
  std::sort(submap_src.begin(), submap_src.end());
  submap_src.erase(std::unique(submap_src.begin(), submap_src.end()),
                   submap_src.end());
  submap_src.shrink_to_fit();

  // Compute submap destination ranks
  // NOTE: The call to NBX can be avoided by a two-step communication process
  // TODO: Can NBX call be avoided by using O^r_p and O^g_p?
  std::vector<int> submap_dest
      = dolfinx::MPI::compute_graph_edges_nbx(comm, submap_src);
  std::sort(submap_dest.begin(), submap_dest.end());

  // Compute the global indices (w.r.t. the submap) of the submap ghosts
  std::vector<std::int64_t> submap_ghost_global(submap_ghost.size());
  imap.local_to_global(submap_ghost, submap_ghost_global);
  auto submap_ghost_gidxs = compute_submap_ghost_indices(
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

  return {IndexMap(comm, submap_local_size, submap_ghost_gidxs,
                   submap_ghost_owners),
          std::move(sub_imap_to_imap)};
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm comm, std::int32_t local_size)
    : _comm(comm, true), _overlapping(false)
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
      _dest(src_dest[1]), _overlapping(true)
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
const std::vector<std::int64_t>& IndexMap::ghosts() const noexcept
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

  std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts(
      _ghosts.size());
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    global_local_ghosts[i] = {_ghosts[i], i + local_size};
  std::map<std::int64_t, std::int32_t> global_to_local(
      global_local_ghosts.begin(), global_local_ghosts.end());

  std::transform(global.begin(), global.end(), local.begin(),
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
MPI_Comm IndexMap::comm() const { return _comm.comm(); }
//----------------------------------------------------------------------------
std::pair<IndexMap, std::vector<std::int32_t>>
IndexMap::create_submap(std::span<const std::int32_t> indices) const
{
  if (!indices.empty() and indices.back() >= this->size_local())
  {
    throw std::runtime_error(
        "Unowned index detected when creating sub-IndexMap");
  }

  // --- Step 1: Compute new offset for this rank

  std::int64_t local_size_new = indices.size();
  std::int64_t offset_new = 0;
  MPI_Request request_offset;
  int ierr = MPI_Iexscan(&local_size_new, &offset_new, 1, MPI_INT64_T, MPI_SUM,
                         _comm.comm(), &request_offset);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  // --- Step 2: Send ghost indices to owning rank
  const std::vector<int>& src = this->src();
  const std::vector<int>& dest = this->dest();
  auto [send_indices, recv_indices, ghost_buffer_pos, send_sizes, recv_sizes,
        send_disp, recv_disp]
      = communicate_ghosts_to_owners(
          this->comm(), src, dest, this->ghosts(), this->owners(),
          std::vector<int>(this->ghosts().size(), 1));

  // --- Step 3: Check which received indexes (all of which I should
  // own) are in the submap

  // Build array for each received ghost that (i) contains the new
  // submap index if it is retained, or (ii) set to -1 if it is not
  // retained.
  std::vector<std::int64_t> send_gidx;
  send_gidx.reserve(recv_indices.size());
  for (auto idx : recv_indices)
  {
    assert(idx - _local_range[0] >= 0);
    assert(idx - _local_range[0] < _local_range[1]);
    std::int32_t idx_local = idx - _local_range[0];

    // Could avoid search by creating look-up array
    auto it = std::lower_bound(indices.begin(), indices.end(), idx_local);
    if (it != indices.end() and *it == idx_local)
    {
      std::size_t idx_local_new = std::distance(indices.begin(), it);
      send_gidx.push_back(idx_local_new + offset_new);
    }
    else
      send_gidx.push_back(-1);
  }

  // --- Step 4: Send new global indices from owner back to ranks that
  // ghost the index

  // Create neighbourhood comm (owner -> ghost)
  MPI_Comm comm1;
  ierr = MPI_Dist_graph_create_adjacent(
      _comm.comm(), src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
      dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  // Send index markers to ghosting ranks
  std::vector<std::int64_t> recv_gidx(send_disp.back());
  ierr = MPI_Neighbor_alltoallv(send_gidx.data(), recv_sizes.data(),
                                recv_disp.data(), MPI_INT64_T, recv_gidx.data(),
                                send_sizes.data(), send_disp.data(),
                                MPI_INT64_T, comm1);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  ierr = MPI_Comm_free(&comm1);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  // --- Step 5: Unpack received data

  std::vector<std::int64_t> ghosts;
  std::vector<int> src_ranks;
  std::vector<std::int32_t> new_to_old_ghost;
  for (std::size_t i = 0; i < send_disp.size() - 1; ++i)
  {
    for (int j = send_disp[i]; j < send_disp[i + 1]; ++j)
    {
      if (std::int64_t idx = recv_gidx[j]; idx >= 0)
      {
        std::size_t p = ghost_buffer_pos[j];
        ghosts.push_back(idx);
        src_ranks.push_back(src[i]);
        new_to_old_ghost.push_back(p);
      }
    }
  }

  if (_overlapping)
  {
    return {IndexMap(_comm.comm(), local_size_new, ghosts, src_ranks),
            std::move(new_to_old_ghost)};
  }
  else
  {
    assert(new_to_old_ghost.empty());
    return {IndexMap(_comm.comm(), local_size_new),
            std::vector<std::int32_t>()};
  }
}
//-----------------------------------------------------------------------------
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
        auto it1 = std::find_if(
            it, idxpos_to_rank.end(),
            [i](auto x) { return x.first != static_cast<std::int32_t>(i); });
        offsets.push_back(offsets.back() + std::distance(it, it1));
        it = it1;
      }
    }
  }

  // Convert ranks for owned indices from neighbour to global ranks
  std::transform(idx_to_rank.begin(), idx_to_rank.end(), data.begin(),
                 [&dest](auto x) { return dest[x.second]; });

  return graph::AdjacencyList<int>(std::move(data), std::move(offsets));
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
const std::vector<int>& IndexMap::src() const noexcept { return _src; }
//-----------------------------------------------------------------------------
const std::vector<int>& IndexMap::dest() const noexcept { return _dest; }
//-----------------------------------------------------------------------------
bool IndexMap::overlapped() const noexcept { return _overlapping; }
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
  MPI_Allreduce(local_sizes.data(), max_count.data(), 2,
                dolfinx::MPI::mpi_type<std::int32_t>(), MPI_MAX, _comm.comm());

  std::int32_t total_num_ghosts = 0;
  MPI_Allreduce(&local_sizes[1], &total_num_ghosts, 1,
                dolfinx::MPI::mpi_type<std::int32_t>(), MPI_SUM, _comm.comm());

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
