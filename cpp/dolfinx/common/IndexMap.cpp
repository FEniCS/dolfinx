// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMapNew.h"
#include <algorithm>
#include <dolfinx/common/sort.h>
#include <functional>
#include <numeric>

using namespace dolfinx;
using namespace dolfinx::common;

namespace
{
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
IndexMapOld::IndexMapOld(MPI_Comm comm, std::int32_t local_size,
                         const xtl::span<const int>& dest_ranks,
                         const xtl::span<const std::int64_t>& ghosts,
                         const xtl::span<const int>& src_ranks)
    : _comm(comm), _comm_owner_to_ghost(MPI_COMM_NULL),
      _comm_ghost_to_owner(MPI_COMM_NULL),
      _ghosts(ghosts.begin(), ghosts.end()),
      _owners(src_ranks.begin(), src_ranks.end())
{
  assert(size_t(ghosts.size()) == src_ranks.size());

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
std::array<std::int64_t, 2> IndexMapOld::local_range() const noexcept
{
  return _local_range;
}
//-----------------------------------------------------------------------------
std::int32_t IndexMapOld::num_ghosts() const noexcept { return _ghosts.size(); }
//-----------------------------------------------------------------------------
std::int32_t IndexMapOld::size_local() const noexcept
{
  return _local_range[1] - _local_range[0];
}
//-----------------------------------------------------------------------------
std::int64_t IndexMapOld::size_global() const noexcept { return _size_global; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& IndexMapOld::ghosts() const noexcept
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>&
IndexMapOld::scatter_fwd_indices() const
{
  if (!_shared_indices)
    throw std::runtime_error("Ooops");
  assert(_shared_indices);
  return *_shared_indices;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>&
IndexMapOld::scatter_fwd_ghost_positions() const noexcept
{
  return _ghost_pos_recv_fwd;
}
//-----------------------------------------------------------------------------
std::vector<int> IndexMapOld::ghost_owners() const
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
MPI_Comm IndexMapOld::comm() const { return _comm.comm(); }
//----------------------------------------------------------------------------
MPI_Comm IndexMapOld::comm(Direction dir) const
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
