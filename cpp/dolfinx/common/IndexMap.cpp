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
      _comm_ghost_to_owner(MPI_COMM_NULL),
      _ghosts(ghosts.begin(), ghosts.end()),
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
const graph::AdjacencyList<std::int32_t>& IndexMap::scatter_fwd_indices() const
{
  if (!_shared_indices)
    throw std::runtime_error("Ooops");
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
std::vector<int> IndexMap::ghost_owners() const
{
  std::vector<int> owners(_ghost_pos_recv_fwd.size());
  std::transform(
      _ghost_pos_recv_fwd.begin(), _ghost_pos_recv_fwd.end(), owners.begin(),
      [&displs = _displs_recv_fwd](auto pos)
      {
        auto it = std::upper_bound(displs.begin(), displs.end(), pos);
        return std::distance(displs.begin(), it) - 1;
      });

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
  std::vector<int> fwd_sharing_offsets = {0};
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

  std::vector<int> ghost_owners;
  {
    std::vector<int> neighbors
        = dolfinx::MPI::neighbors(_comm_owner_to_ghost.comm())[0];
    ghost_owners = this->ghost_owners();
    std::transform(ghost_owners.cbegin(), ghost_owners.cend(),
                   ghost_owners.begin(),
                   [&neighbors](auto r) { return neighbors[r]; });
  }

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
        shared_indices[idx].insert(recv_data[i + 2 + j]);
    }
    i += set_size + 2;
  }

  return shared_indices;
}
//-----------------------------------------------------------------------------
