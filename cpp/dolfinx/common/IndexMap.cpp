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

#include <iostream>

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
} // namespace

//-----------------------------------------------------------------------------
std::vector<int32_t>
common::compute_owned_indices(std::span<const std::int32_t> indices,
                              const IndexMap& map)
{
  // Build list of (owner, index) pairs for each ghost in indices, and
  // sort
  std::vector<std::pair<int, std::int64_t>> send_idx;
  std::for_each(indices.begin(), indices.end(),
                [&send_idx, &owners = map.owners(), &ghosts = map.ghosts(),
                 size = map.size_local()](auto idx)
                {
                  if (idx >= size)
                  {
                    std::int32_t pos = idx - size;
                    send_idx.push_back({owners[pos], ghosts[pos]});
                  }
                });
  std::sort(send_idx.begin(), send_idx.end());

  // Build (i) list of src ranks, (ii) send buffer, (iii) send sizes and
  // (iv) send displacements
  std::vector<int> src;
  std::vector<std::int64_t> send_buffer;
  std::vector<int> send_sizes, send_disp(1, 0);
  auto it = send_idx.begin();
  while (it != send_idx.end())
  {
    src.push_back(it->first);
    auto it1
        = std::find_if(it, send_idx.end(),
                       [r = src.back()](auto& idx) { return idx.first != r; });

    // Pack send buffer
    std::transform(it, it1, std::back_inserter(send_buffer),
                   [](auto& idx) { return idx.second; });

    // Send sizes and displacements
    send_sizes.push_back(std::distance(it, it1));
    send_disp.push_back(send_disp.back() + send_sizes.back());

    // Advance iterator
    it = it1;
  }

  // Determine destination ranks
  std::vector<int> dest
      = dolfinx::MPI::compute_graph_edges_nbx(map.comm(), src);
  std::sort(dest.begin(), dest.end());

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
  std::copy_if(indices.begin(), indices.end(), std::back_inserter(owned),
               [size = map.size_local()](auto idx) { return idx < size; });
  std::transform(recv_buffer.begin(), recv_buffer.end(),
                 std::back_inserter(owned),
                 [range = map.local_range()](auto idx)
                 {
                   assert(idx >= range[0]); // problem
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
  std::vector<int> src;
  for (auto& map : maps)
  {
    src.insert(src.end(), map.first.get().owners().begin(),
               map.first.get().owners().end());
    std::sort(src.begin(), src.end());
    src.erase(std::unique(src.begin(), src.end()), src.end());
  }

  // Get destination ranks (ranks that ghost my indices), and sort
  std::vector<int> dest = dolfinx::MPI::compute_graph_edges_nbx(
      maps.at(0).first.get().comm(), src);
  std::sort(dest.begin(), dest.end());

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
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm comm, std::int32_t local_size)
    : _comm(comm), _overlapping(false)
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
    : _comm(comm), _ghosts(ghosts.begin(), ghosts.end()),
      _owners(owners.begin(), owners.end()), _src(src_dest[0]),
      _dest(src_dest[1]), _overlapping(true)
{
  assert(ghosts.size() == owners.size());
  assert(std::is_sorted(src_dest[0].begin(), src_dest[0].end()));
  assert(std::is_sorted(src_dest[1].begin(), src_dest[1].end()));

  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = (std::int64_t)local_size;
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

  // Build list of src ranks (ranks that own ghosts)
  std::vector<int> src = this->owners();
  std::sort(src.begin(), src.end());
  src.erase(std::unique(src.begin(), src.end()), src.end());

  // Determine destination ranks (ranks that ghost my indices), and sort
  std::vector<int> dest
      = dolfinx::MPI::compute_graph_edges_nbx(this->comm(), src);
  std::sort(dest.begin(), dest.end());

  std::vector<std::int64_t> recv_indices;
  std::vector<std::size_t> ghost_buffer_pos;
  std::vector<int> send_disp, recv_disp;
  std::vector<std::int32_t> send_sizes, recv_sizes;
  {
    // Create neighbourhood comm (ghost -> owner)
    MPI_Comm comm0;
    int ierr = MPI_Dist_graph_create_adjacent(
        _comm.comm(), dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(),
        src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    dolfinx::MPI::check_error(_comm.comm(), ierr);

    // Pack ghosts indices
    std::vector<std::vector<std::int64_t>> send_data(src.size());
    std::vector<std::vector<std::size_t>> pos_to_ghost(src.size());
    for (std::size_t i = 0; i < _ghosts.size(); ++i)
    {
      auto it = std::lower_bound(src.begin(), src.end(), _owners[i]);
      assert(it != src.end() and *it == _owners[i]);
      int r = std::distance(src.begin(), it);
      send_data[r].push_back(_ghosts[i]);
      pos_to_ghost[r].push_back(i);
    }

    // Count number of ghosts per dest
    std::transform(send_data.begin(), send_data.end(),
                   std::back_inserter(send_sizes),
                   [](auto& d) { return d.size(); });

    // Build send buffer and ghost position to send buffer position
    std::vector<std::int64_t> send_indices;
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
    dolfinx::MPI::check_error(_comm.comm(), ierr);

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
    dolfinx::MPI::check_error(_comm.comm(), ierr);

    ierr = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(_comm.comm(), ierr);
  }

  ierr = MPI_Wait(&request_offset, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

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
std::pair<std::vector<std::int32_t>,
          std::pair<IndexMap, std::vector<std::int32_t>>>
IndexMap::create_submap(
    const std::span<const std::int32_t>& indices,
    const std::span<const std::int32_t>& connected_indices) const
{
  const int rank = MPI::rank(comm());

  if (!indices.empty() and indices.back() >= this->size_local())
  {
    throw std::runtime_error(
        "Unowned index detected when creating sub-IndexMap");
  }

  // Create a list of indices that are owned and connected on this process
  std::vector<std::int32_t> owned_connected_indices;
  std::set_intersection(indices.begin(), indices.end(),
                        connected_indices.begin(), connected_indices.end(),
                        std::back_inserter(owned_connected_indices));

  // Lookup array to determine if an owned local index is in the submap
  std::vector<bool> in_submap(size_local(), false);
  for (std::int32_t i : indices)
    in_submap[i] = true;

  // Lookup array to determine if a local index is connected on this process
  std::vector<bool> is_connected(size_local() + num_ghosts(), false);
  for (std::int32_t i : connected_indices)
    is_connected[i] = true;

  // --- Step 1: Send ghost indices to owning rank. Also, send array
  // marking ghosts that are connected on this process.

  // Build list of src ranks (ranks that own ghosts)
  std::vector<int> src = this->owners();
  std::sort(src.begin(), src.end());
  src.erase(std::unique(src.begin(), src.end()), src.end());

  // Determine destination ranks (ranks that ghost my indices), and sort
  std::vector<int> dest
      = dolfinx::MPI::compute_graph_edges_nbx(this->comm(), src);
  std::sort(dest.begin(), dest.end());

  // Ghost indices on this process
  std::vector<std::int64_t> ghost_indices_send;
  // Marker for ghosts on this process that are connected on this process
  // (1 if connected, 0 if unconnected)
  // TODO Make this a bool array
  std::vector<std::int32_t> ghost_connected_indices_send;
  //  Indices owned by this process that are ghosted by other processes
  std::vector<std::int64_t> ghost_indices_recv;
  // Marker for indices owned by this process determining if they are
  // connected on the processes that ghost them (1 if connected, 0 if
  // unconnected)
  std::vector<std::int32_t> ghost_connected_indices_recv;
  std::vector<std::size_t> ghost_buffer_pos;
  std::vector<int> ghost_send_disp, ghost_recv_disp;
  std::vector<std::int32_t> ghost_send_sizes, ghost_recv_sizes;

  // Create neighbourhood comm (ghost -> owner)
  MPI_Comm comm0;
  MPI_Dist_graph_create_adjacent(_comm.comm(), dest.size(), dest.data(),
                                 MPI_UNWEIGHTED, src.size(), src.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);

  // Pack ghosts indices
  std::vector<std::vector<std::int64_t>> send_data(src.size());
  std::vector<std::vector<std::size_t>> pos_to_ghost(src.size());
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
  {
    auto it = std::lower_bound(src.begin(), src.end(), _owners[i]);
    assert(it != src.end() and *it == _owners[i]);
    int r = std::distance(src.begin(), it);
    send_data[r].push_back(_ghosts[i]);
    pos_to_ghost[r].push_back(i);
  }

  // Count number of ghosts per dest
  std::transform(send_data.begin(), send_data.end(),
                 std::back_inserter(ghost_send_sizes),
                 [](auto& d) { return d.size(); });

  // Build send buffer and ghost position to send buffer position
  for (auto& d : send_data)
    ghost_indices_send.insert(ghost_indices_send.end(), d.begin(), d.end());
  for (auto& p : pos_to_ghost)
    ghost_buffer_pos.insert(ghost_buffer_pos.end(), p.begin(), p.end());

  // Convert to local to check if the indices are connected on this process
  std::vector<std::int32_t> ghost_indices_send_local(ghost_indices_send.size());
  global_to_local(ghost_indices_send, ghost_indices_send_local);
  // Mark the indices ghosted by this process that are connected on this
  // process
  for (std::int32_t ghost_index_local : ghost_indices_send_local)
  {
    // Mark 1 if connected else 0
    ghost_connected_indices_send.push_back(is_connected[ghost_index_local]);
  }

  // Send how many indices I ghost to each owner, and receive how many
  // of my indices other ranks ghost
  ghost_recv_sizes.resize(dest.size(), 0);
  ghost_send_sizes.reserve(1);
  ghost_recv_sizes.reserve(1);
  MPI_Neighbor_alltoall(ghost_send_sizes.data(), 1, MPI_INT32_T,
                        ghost_recv_sizes.data(), 1, MPI_INT32_T, comm0);

  // Prepare displacement vectors
  ghost_send_disp.resize(src.size() + 1, 0);
  ghost_recv_disp.resize(dest.size() + 1, 0);
  std::partial_sum(ghost_send_sizes.begin(), ghost_send_sizes.end(),
                   std::next(ghost_send_disp.begin()));
  std::partial_sum(ghost_recv_sizes.begin(), ghost_recv_sizes.end(),
                   std::next(ghost_recv_disp.begin()));

  // Send ghost indices to owner, and receive indices
  ghost_indices_recv.resize(ghost_recv_disp.back());
  MPI_Neighbor_alltoallv(ghost_indices_send.data(), ghost_send_sizes.data(),
                         ghost_send_disp.data(), MPI_INT64_T,
                         ghost_indices_recv.data(), ghost_recv_sizes.data(),
                         ghost_recv_disp.data(), MPI_INT64_T, comm0);

  ghost_connected_indices_recv.resize(ghost_recv_disp.back());
  MPI_Neighbor_alltoallv(
      ghost_connected_indices_send.data(), ghost_send_sizes.data(),
      ghost_send_disp.data(), MPI_INT32_T, ghost_connected_indices_recv.data(),
      ghost_recv_sizes.data(), ghost_recv_disp.data(), MPI_INT32_T, comm0);

  // --- Step 2: For each index I own which are ghosted by other ranks,
  // figure out who the new owner should be and tell them

  // Create a map from each of the indices that I own that are ghosted on
  // other processes to processes that could possibly become their new
  // owner (i.e. processes on which they are connected)
  // TODO Try to replace map
  std::map<std::int64_t, std::vector<std::int32_t>> possible_new_owners;
  // Loop over destination ranks
  for (std::size_t i = 0; i < dest.size(); ++i)
  {
    // Loop through indices I own that are ghosted by process dest[i]
    for (int j = ghost_recv_disp[i]; j < ghost_recv_disp[i + 1]; ++j)
    {
      std::int64_t global_index = ghost_indices_recv[j];
      // If this index is connected on process dest[i], add it to the list
      // of possible new owners
      if (ghost_connected_indices_recv[j] == 1)
      {
        possible_new_owners[global_index].push_back(dest[i]);
      }
    }
  }

  // Map from global index of indices I own that are shared to their
  // new owner
  // TODO Replace map
  std::map<std::int64_t, std::int32_t> new_owners;
  for (auto [global_index, possible_owners] : possible_new_owners)
  {
    // Not all owned shared indices will be in the submap, and some
    // of those that are will be connected on this rank and but not
    // on the ranks that ghost them, so check we need to check the
    // size of `possible_owners`
    if (possible_owners.size() > 0)
    {
      // Choose new owner randomly for load balancing
      const int random_index = std::rand() % possible_owners.size();
      new_owners[global_index] = possible_owners[random_index];
    }
  }

  // Determine the new owner of indices I own that are ghosted on other
  // processes. If the index is connected on this process, I remain its
  // owner. Otherwise, determine the new owner from `possible_new_owners`
  // and communicate this to other processes.

  // Storage for new owners
  std::vector<std::int32_t> new_owners_send;
  new_owners_send.reserve(ghost_indices_recv.size());
  // Loop through indices I own that are ghosted on other processes
  for (std::size_t i = 0; i < ghost_indices_recv.size(); ++i)
  {
    std::int64_t global_index = ghost_indices_recv[i];
    assert(global_index - _local_range[0] >= 0);
    assert(global_index - _local_range[0] < _local_range[1]);
    std::int32_t local_index = global_index - _local_range[0];

    // Check if the index is in the submap. If so, determine its owner,
    // else mark with -1.
    if (in_submap[local_index])
    {
      // See if this index is connected on this process. If so, I continue to
      // own this index. Otherwise, use the new owner.
      if (is_connected[local_index])
      {
        new_owners_send.push_back(rank);
      }
      else
      {
        new_owners_send.push_back(new_owners[global_index]);
      }
    }
    else
    {
      new_owners_send.push_back(-1);
    }
  }

  // Create neighbourhood comm (owner -> ghost)
  MPI_Comm comm1;
  MPI_Dist_graph_create_adjacent(_comm.comm(), src.size(), src.data(),
                                 MPI_UNWEIGHTED, dest.size(), dest.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);

  // Send who the new owners of indices I currently own are. Receive who
  // the new owners of indices I ghost are
  std::vector<std::int32_t> new_owners_recv(ghost_send_disp.back());
  MPI_Neighbor_alltoallv(new_owners_send.data(), ghost_recv_sizes.data(),
                         ghost_recv_disp.data(), MPI_INT32_T,
                         new_owners_recv.data(), ghost_send_sizes.data(),
                         ghost_send_disp.data(), MPI_INT32_T, comm1);

  // --- Step 3: Take ownership of new indices and compute their new
  // global index in the submap. Send the new global index back to
  // their original owner.
  // NOTE: This rank computes the new global index of indices it has
  // taken ownership of, and sends it back to the original owner. The
  // original owner then sends it to the ghosts. This two-step process
  // avoids having to discover the new neighbourhood.

  // Loop through the indices that I ghost. If I am the new owner of that
  // index, add it to `owned_connected_indices`
  for (std::size_t i = 0; i < ghost_indices_send.size(); ++i)
  {
    if (new_owners_recv[i] == rank)
    {
      owned_connected_indices.push_back(ghost_indices_send_local[i]);
    }
  }

  // Compute new offset for this rank
  std::int64_t local_size_new = owned_connected_indices.size();
  std::int64_t offset_new = 0;
  MPI_Request request_offset;
  MPI_Iexscan(&local_size_new, &offset_new, 1, MPI_INT64_T, MPI_SUM,
              _comm.comm(), &request_offset);
  MPI_Wait(&request_offset, MPI_STATUS_IGNORE);

  // Loop through the indices I ghost. If I am the new owner, compute the
  // new global index
  std::vector<std::int64_t> send_gidx_to_original_owner;
  for (std::size_t i = 0; i < ghost_indices_send.size(); ++i)
  {
    if (new_owners_recv[i] == rank)
    {
      auto it = std::find(owned_connected_indices.begin(),
                          owned_connected_indices.end(),
                          ghost_indices_send_local[i]);
      int new_local_idx = std::distance(owned_connected_indices.begin(), it);
      send_gidx_to_original_owner.push_back(new_local_idx + offset_new);
    }
    else
    {
      send_gidx_to_original_owner.push_back(-1);
    }
  }

  // Send the global indices of indices that I am the new owner of two their
  // previous owners. Receive the new global indices of indices I used to own
  std::vector<std::int64_t> recv_gidx_to_original_owner(ghost_recv_disp.back());
  MPI_Neighbor_alltoallv(
      send_gidx_to_original_owner.data(), ghost_send_sizes.data(),
      ghost_send_disp.data(), MPI_INT64_T, recv_gidx_to_original_owner.data(),
      ghost_recv_sizes.data(), ghost_recv_disp.data(), MPI_INT64_T, comm0);
  MPI_Comm_free(&comm0);

  // Create a map from old global index to new global index for the indices
  // I used to own but who's owner has changed
  std::map<std::int64_t, std::int64_t> ownership_change_global_map;
  for (std::size_t i = 0; i < ghost_indices_recv.size(); ++i)
  {
    if (recv_gidx_to_original_owner[i] != -1)
    {
      ownership_change_global_map[ghost_indices_recv[i]]
          = recv_gidx_to_original_owner[i];
    }
  }

  // --- Step 4: Check which received indexes (all of which I own
  // in the original map, but some of which may have new owners
  // in the new map) are in the submap, and send their new global
  // index to the ranks that share them. NOTE: The original owner
  // must send of an index that has changed ownership must send the
  // new global index (which was received from the new owner) to
  // ensure that all ranks ghosting it have the correct new owner

  // Build array for each received ghost that (i) contains the new
  // submap index if it is retained, or (ii) set to -1 if it is not
  // retained.
  std::vector<std::int64_t> send_gidx;
  send_gidx.reserve(ghost_indices_recv.size());
  for (auto idx : ghost_indices_recv)
  {
    assert(idx - _local_range[0] >= 0);
    assert(idx - _local_range[0] < _local_range[1]);
    std::int32_t idx_local = idx - _local_range[0];

    // Could avoid search by creating look-up array
    auto it = std::lower_bound(owned_connected_indices.begin(),
                               owned_connected_indices.end(), idx_local);
    if (it != owned_connected_indices.end() and *it == idx_local)
    {
      // If I own this index and it is connected on this process
      // (this includes indices who's ownership has changed to this
      // process), compute the new global index to send to other
      // processes ghosting this index
      std::size_t idx_local_new
          = std::distance(owned_connected_indices.begin(), it);
      send_gidx.push_back(idx_local_new + offset_new);
    }
    else if (ownership_change_global_map.contains(idx))
    {
      // Else, if I used to own this index, but it is not connected
      // on this process and therefore has changed owner, add the
      // new index (sent from the new owner) so that other processes
      // ghosting this index know the new global index
      send_gidx.push_back(ownership_change_global_map.at(idx));
    }
    else
    {
      // Else, the ghost index is not in the submap
      send_gidx.push_back(-1);
    }
  }

  // --- Step 5: Send new global indices from owner back to ranks that
  // ghost the index. Receive the new global indices of the indices I
  // ghost

  // New (submap) global index of the indices I ghost
  std::vector<std::int64_t> recv_gidx(ghost_send_disp.back());
  MPI_Neighbor_alltoallv(send_gidx.data(), ghost_recv_sizes.data(),
                         ghost_recv_disp.data(), MPI_INT64_T, recv_gidx.data(),
                         ghost_send_sizes.data(), ghost_send_disp.data(),
                         MPI_INT64_T, comm1);

  MPI_Comm_free(&comm1);

  // --- Step 6: Unpack received data

  std::vector<std::int64_t> ghosts;
  std::vector<int> src_ranks;
  // Create a map from the old (original index map) ghost position to the
  // new (submap) ghost position
  std::vector<std::int32_t> new_to_old_ghost;
  // TODO Change to src.size()
  for (std::size_t i = 0; i < src.size(); ++i)
  {
    for (int j = ghost_send_disp[i]; j < ghost_send_disp[i + 1]; ++j)
    {
      // Only add the ghost index if it is (i) in the submap, (ii) connected
      // on this process, and (iii) I have not taken ownership of it from
      // another process
      if (std::int64_t idx = recv_gidx[j];
          idx >= 0 and is_connected[ghost_indices_send_local[j]]
          and idx != send_gidx_to_original_owner[j])
      {
        std::size_t p = ghost_buffer_pos[j];
        ghosts.push_back(idx);
        src_ranks.push_back(new_owners_recv[j]);
        new_to_old_ghost.push_back(p);
      }
    }
  }

  if (_overlapping)
  {
    return std::make_pair(std::move(owned_connected_indices),
                          std::make_pair(IndexMap(_comm.comm(), local_size_new,
                                                  ghosts, src_ranks),
                                         std::move(new_to_old_ghost)));
  }
  else
  {
    assert(new_to_old_ghost.empty());
    return std::make_pair(std::move(owned_connected_indices),
                          std::make_pair(IndexMap(_comm.comm(), local_size_new),
                                         std::vector<std::int32_t>()));
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
  // Build list of (owner, index) pairs for each ghost, and sort
  std::vector<std::pair<int, std::int64_t>> send_idx;
  std::transform(_ghosts.begin(), _ghosts.end(), _owners.begin(),
                 std::back_inserter(send_idx),
                 [](auto idx, auto r)
                 { return std::pair<int, std::int64_t>(r, idx); });
  std::sort(send_idx.begin(), send_idx.end());

  std::vector<int> src;
  std::vector<std::int64_t> send_buffer;
  std::vector<int> send_sizes, send_disp{0};
  {
    auto it = send_idx.begin();
    while (it != send_idx.end())
    {
      src.push_back(it->first);
      auto it1 = std::find_if(it, send_idx.end(),
                              [r = src.back()](auto& idx)
                              { return idx.first != r; });

      // Pack send buffer
      std::transform(it, it1, std::back_inserter(send_buffer),
                     [](auto& idx) { return idx.second; });

      // Send sizes and displacements
      send_sizes.push_back(std::distance(it, it1));
      send_disp.push_back(send_disp.back() + send_sizes.back());

      // Advance iterator
      it = it1;
    }
  }

  auto dest = dolfinx::MPI::compute_graph_edges_nbx(_comm.comm(), src);
  std::sort(dest.begin(), dest.end());

  // Create ghost -> owner comm
  MPI_Comm comm;
  int ierr = MPI_Dist_graph_create_adjacent(
      _comm.comm(), dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(),
      src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
  dolfinx::MPI::check_error(_comm.comm(), ierr);

  std::vector<int> recv_sizes(dest.size(), 0);
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(),
                               1, MPI_INT, comm);
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
