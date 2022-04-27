// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMapNew.h"
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
} // namespace

//-----------------------------------------------------------------------------
common::IndexMap common::create_old(const IndexMapNew& map)
{
  std::vector<int> src_ranks = map.owners();
  std::sort(src_ranks.begin(), src_ranks.end());
  src_ranks.erase(std::unique(src_ranks.begin(), src_ranks.end()),
                  src_ranks.end());

  auto dest_ranks
      = dolfinx::MPI::compute_graph_edges_nbx(map.comm(), src_ranks);
  return IndexMap(map.comm(), map.size_local(), dest_ranks, map.ghosts(),
                  map.owners());
}
//-----------------------------------------------------------------------------
common::IndexMapNew common::create_new(const IndexMap& map)
{
  return IndexMapNew(map.comm(), map.size_local(), map.ghosts(), map.owners());
}
//-----------------------------------------------------------------------------
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
common::stack_index_maps(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMapNew>, int>>&
        maps)
{
  // Compute process offset for stacked index map
  const std::int64_t process_offset = std::accumulate(
      maps.begin(), maps.end(), std::int64_t(0),
      [](std::int64_t c, auto& map) -> std::int64_t
      { return c + map.first.get().local_range()[0] * map.second; });

  // Get local offset (into new map) for each index map
  std::vector<std::int32_t> local_offset(maps.size() + 1, 0);
  for (std::size_t f = 1; f < local_offset.size(); ++f)
  {
    std::int32_t local_size = maps[f - 1].first.get().size_local();
    int bs = maps[f - 1].second;
    local_offset[f] = local_offset[f - 1] + bs * local_size;
  }

  // Build list of src ranks
  std::vector<int> src;
  for (auto& map : maps)
  {
    // Get owning ranks
    src.insert(src.end(), map.first.get().owners().begin(),
               map.first.get().owners().end());
    std::sort(src.begin(), src.end());
    src.erase(std::unique(src.begin(), src.end()), src.end());
  }

  // Get destination ranks, and sort
  std::vector<int> dest = dolfinx::MPI::compute_graph_edges_nbx(
      maps.at(0).first.get().comm(), src);
  std::sort(dest.begin(), dest.end());

  // Create neighbour comm (ghost -> owner)
  MPI_Comm comm0;
  MPI_Dist_graph_create_adjacent(
      maps.at(0).first.get().comm(), dest.size(), dest.data(), MPI_UNWEIGHTED,
      src.size(), src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);

  // Create neighbour comm (owner -> ghost)
  MPI_Comm comm1;
  MPI_Dist_graph_create_adjacent(
      maps.at(0).first.get().comm(), src.size(), src.data(), MPI_UNWEIGHTED,
      dest.size(), dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);

  // Send number of ghosts per map to owning rank
  std::vector<std::vector<std::int64_t>> ghosts_new(maps.size());
  std::vector<std::vector<int>> ghost_owners_new(maps.size());
  for (std::size_t m = 0; m < maps.size(); ++m)
  {
    const common::IndexMapNew& map = maps[m].first.get();
    const std::vector<std::int64_t>& ghosts = map.ghosts();
    const std::vector<int>& owners = map.owners();

    // For each owning rank (on comm), create vector of this rank's
    // ghosts
    std::vector<std::int64_t> send_indices;
    std::vector<std::int32_t> send_sizes;
    std::vector<std::size_t> ghost_idx_buffer;
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
                     std::back_insert_iterator(send_sizes),
                     [](auto& g) { return g.size(); });

      // Send buffer
      for (auto& g : ghost_by_rank)
        send_indices.insert(send_indices.end(), g.begin(), g.end());

      for (auto& p : pos_to_ghost)
        ghost_idx_buffer.insert(ghost_idx_buffer.end(), p.begin(), p.end());
    }

    // Send how many indices I ghost to each owner, and receive data
    // from ranks that ghost my indices
    std::vector<std::int32_t> recv_sizes(dest.size(), 0);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT32_T, recv_sizes.data(),
                          1, MPI_INT32_T, comm0);

    // Send ghost indices to owner
    std::vector<int> send_disp(src.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::next(send_disp.begin()));

    std::vector<int> recv_disp(dest.size() + 1, 0);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::next(recv_disp.begin()));

    std::vector<std::int64_t> recv_indices(recv_disp.back());
    MPI_Neighbor_alltoallv(send_indices.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, recv_indices.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           comm0);

    // For each received ghost, compute new index
    std::vector<std::int64_t> ghost_old_to_new;
    ghost_old_to_new.reserve(recv_indices.size());
    std::int64_t offset_old = map.local_range()[0];
    std::int64_t offset_new = local_offset[m] + process_offset;
    for (std::int64_t idx : recv_indices)
    {
      auto idx_local = idx - offset_old;
      assert(idx_local >= 0);
      ghost_old_to_new.push_back(idx_local + offset_new);
    }

    // Send back new indices
    std::vector<std::int64_t> ghosts_new_idx(send_disp.back());
    MPI_Neighbor_alltoallv(ghost_old_to_new.data(), recv_sizes.data(),
                           recv_disp.data(), MPI_INT64_T, ghosts_new_idx.data(),
                           send_sizes.data(), send_disp.data(), MPI_INT64_T,
                           comm1);

    ghosts_new[m].resize(map.ghosts().size());
    for (std::size_t j = 0; j < ghosts_new_idx.size(); ++j)
    {
      std::size_t p = ghost_idx_buffer[j];
      ghosts_new[m][p] = ghosts_new_idx[j];
    }

    ghost_owners_new[m].resize(map.ghosts().size());
    for (std::size_t i = 0; i < recv_disp.size() - 1; ++i)
    {
      int rank = src[i];
      for (int k = recv_disp[i]; k < recv_disp[i + 1]; ++k)
      {
        std::size_t p = ghost_idx_buffer[i];
        ghost_owners_new[m][p] = rank;
      }
    }
  }

  // Destroy communicators
  MPI_Comm_free(&comm0);
  MPI_Comm_free(&comm1);

  return {process_offset, std::move(local_offset), std::move(ghosts_new),
          std::move(ghost_owners_new)};

  // return {process_offset, std::vector<std::int32_t>(),
  //         std::vector<std::vector<std::int64_t>>(),
  //         std::vector<std::vector<int>>()};
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
IndexMapNew::IndexMapNew(MPI_Comm comm, std::int32_t local_size) : _comm(comm)
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
}
//-----------------------------------------------------------------------------
IndexMapNew::IndexMapNew(MPI_Comm comm, std::int32_t local_size,
                         const xtl::span<const std::int64_t>& ghosts,
                         const xtl::span<const int>& src_ranks)
    : _comm(comm), _ghosts(ghosts.begin(), ghosts.end()),
      _owners(src_ranks.begin(), src_ranks.end())
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

  // Wait for MPI_Iexscan to complete (get offset)
  MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  _local_range = {offset, offset + local_size};

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMapNew::local_range() const noexcept
{
  return _local_range;
}
//-----------------------------------------------------------------------------
std::int32_t IndexMapNew::num_ghosts() const noexcept { return _ghosts.size(); }
//-----------------------------------------------------------------------------
std::int32_t IndexMapNew::size_local() const noexcept
{
  return _local_range[1] - _local_range[0];
}
//-----------------------------------------------------------------------------
std::int64_t IndexMapNew::size_global() const noexcept { return _size_global; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& IndexMapNew::ghosts() const noexcept
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
void IndexMapNew::local_to_global(const xtl::span<const std::int32_t>& local,
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
void IndexMapNew::global_to_local(const xtl::span<const std::int64_t>& global,
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
std::vector<std::int64_t> IndexMapNew::global_indices() const
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
MPI_Comm IndexMapNew::comm() const { return _comm.comm(); }
//----------------------------------------------------------------------------
