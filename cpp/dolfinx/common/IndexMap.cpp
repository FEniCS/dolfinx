// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include <algorithm>
#include <numeric>

using namespace dolfinx;
using namespace dolfinx::common;

namespace
{
//-----------------------------------------------------------------------------
void local_to_global_impl(
    Eigen::Ref<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> global,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&
        indices,
    const std::int64_t global_offset, const std::int32_t local_size,
    const int block_size,
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts, bool blocked)
{
  if (blocked)
  {
    for (Eigen::Index i = 0; i < indices.rows(); ++i)
    {
      if (indices[i] < local_size)
        global[i] = global_offset + indices[i];
      else
      {
        assert((indices[i] - local_size) < ghosts.size());
        global[i] = ghosts[indices[i] - local_size];
      }
    }
  }
  else
  {
    for (Eigen::Index i = 0; i < indices.rows(); ++i)
    {
      const std::int32_t index_block = indices[i] / block_size;
      const std::int32_t pos = indices[i] % block_size;
      if (index_block < local_size)
        global[i] = block_size * (global_offset + index_block) + pos;
      else
      {
        assert((index_block - local_size) < ghosts.size());
        global[i] = block_size * ghosts[index_block - local_size] + pos;
      }
    }
  }
}
//-----------------------------------------------------------------------------

/// Compute the owning rank of ghost indices
std::vector<int> get_ghost_ranks(
    MPI_Comm comm, std::int32_t local_size,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
        ghosts)
{
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  std::vector<std::int32_t> local_sizes(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                comm);

  std::vector<std::int64_t> all_ranges(mpi_size + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   all_ranges.begin() + 1);

  // Compute rank of ghost owners
  std::vector<int> ghost_ranks(ghosts.size(), -1);
  for (int i = 0; i < ghosts.size(); ++i)
  {
    auto it = std::upper_bound(all_ranges.begin(), all_ranges.end(), ghosts[i]);
    const int p = std::distance(all_ranges.begin(), it) - 1;
    ghost_ranks[i] = p;
  }

  return ghost_ranks;
}
//-----------------------------------------------------------------------------
// Compute (owned) global indices shared with neighbor processes
std::tuple<std::vector<std::int64_t>, std::vector<std::int32_t>>
compute_forward_indices(
    MPI_Comm comm, const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& ghost_src_ranks)
{
  auto [neighbors_in, neighbors_out] = dolfinx::MPI::neighbors(comm);
  assert(ghosts.size() == ghost_src_ranks.size());

  std::vector<int> out_edges_num(neighbors_out.size(), 0);
  for (int i = 0; i < ghost_src_ranks.size(); ++i)
  {
    const int owner_rank = ghost_src_ranks[i];
    out_edges_num[owner_rank]++;
  }

  std::vector<int> in_edges_num(neighbors_in.size());
  MPI_Neighbor_alltoall(out_edges_num.data(), 1, MPI_INT, in_edges_num.data(),
                        1, MPI_INT, comm);

  std::vector<int> send_disp(neighbors_out.size() + 1, 0);
  std::vector<int> recv_disp(neighbors_in.size() + 1, 0);

  std::partial_sum(out_edges_num.begin(), out_edges_num.end(),
                   send_disp.begin() + 1);
  std::partial_sum(in_edges_num.begin(), in_edges_num.end(),
                   recv_disp.begin() + 1);

  std::vector<std::int64_t> send_indices(send_disp.back());
  std::vector<int> insert_disp(send_disp);
  for (int i = 0; i < ghosts.size(); ++i)
  {
    const int owner_rank = ghost_src_ranks[i];
    send_indices[insert_disp[owner_rank]] = ghosts[i];
    insert_disp[owner_rank]++;
  }

  // A rank in the neighborhood communicator can have no incoming or
  // outcoming edges. This may cause OpenMPI to crash. Workaround:
  in_edges_num.reserve(1);
  out_edges_num.reserve(1);

  // May have repeated shared indices with different processes
  std::vector<std::int64_t> recv_indices(recv_disp.back());
  MPI_Neighbor_alltoallv(send_indices.data(), out_edges_num.data(),
                         send_disp.data(), MPI_INT64_T, recv_indices.data(),
                         in_edges_num.data(), recv_disp.data(), MPI_INT64_T,
                         comm);

  return {recv_indices, in_edges_num};
}
//-----------------------------------------------------------------------------
/// Create neighbourhood communicators
/// @param[in] comm Communicator create communicators with neighborhood
///   topology from
/// @param[in] halo_src_ranks Ranks that own indices in the halo (ghost
///   region) of the calling rank
/// @param[in] halo_dest_ranks Ranks that have indices owned by the
///   calling process own indices in their halo (ghost region)
std::array<MPI_Comm, 3>
compute_asymmetric_communicators(MPI_Comm comm,
                                 std::vector<int>& halo_src_ranks,
                                 std::vector<int>& halo_dest_ranks)
{
  std::array<MPI_Comm, 3> comms{MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL};

  // Create communicator with edges owner (sources) -> ghost
  // (destinations)
  {
    std::vector<int> sourceweights(halo_src_ranks.size(), 1);
    std::vector<int> destweights(halo_dest_ranks.size(), 1);
    MPI_Dist_graph_create_adjacent(
        comm, halo_src_ranks.size(), halo_src_ranks.data(),
        sourceweights.data(), halo_dest_ranks.size(), halo_dest_ranks.data(),
        destweights.data(), MPI_INFO_NULL, false, &comms[0]);
  }

  // Create communicator with edges ghost (sources) -> owner
  // (destinations)
  {
    std::vector<int> sourceweights(halo_dest_ranks.size(), 1);
    std::vector<int> destweights(halo_src_ranks.size(), 1);
    MPI_Dist_graph_create_adjacent(
        comm, halo_dest_ranks.size(), halo_dest_ranks.data(),
        sourceweights.data(), halo_src_ranks.size(), halo_src_ranks.data(),
        destweights.data(), MPI_INFO_NULL, false, &comms[1]);
  }

  // Create communicator two-way edges
  // TODO: Aim to remove? used for compatibility
  {
    std::vector<int> neighbors;
    std::set_union(halo_dest_ranks.begin(), halo_dest_ranks.end(),
                   halo_src_ranks.begin(), halo_src_ranks.end(),
                   std::back_inserter(neighbors));
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());

    std::vector<int> sourceweights(neighbors.size(), 1);
    std::vector<int> destweights(neighbors.size(), 1);
    MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                   sourceweights.data(), neighbors.size(),
                                   neighbors.data(), destweights.data(),
                                   MPI_INFO_NULL, false, &comms[2]);
  }

  return comms;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
common::stack_index_maps(
    const std::vector<std::reference_wrapper<const common::IndexMap>>& maps)
{

  // Get process offset
  std::int64_t process_offset = 0;
  for (const common::IndexMap& map : maps)
    process_offset += map.local_range()[0] * map.block_size();

  // Get local map offset
  std::vector<std::int32_t> local_offset(maps.size() + 1, 0);
  for (std::size_t f = 1; f < local_offset.size(); ++f)
  {
    local_offset[f]
        = local_offset[f - 1]
          + maps[f - 1].get().size_local() * maps[f - 1].get().block_size();
  }

  // Pack old and new composite indices for owned entries that are ghost
  // on other ranks
  std::vector<std::int64_t> indices;
  for (std::size_t f = 0; f < maps.size(); ++f)
  {
    const int bs = maps[f].get().block_size();
    const std::vector<std::int32_t>& forward_indices
        = maps[f].get().forward_indices();
    const std::int64_t offset = bs * maps[f].get().local_range()[0];
    for (std::int32_t local_index : forward_indices)
    {
      for (std::int32_t i = 0; i < bs; ++i)
      {
        // Insert field index, global index, composite global index
        indices.push_back(f);
        indices.push_back(bs * local_index + i + offset);
        indices.push_back(bs * local_index + i + local_offset[f]
                          + process_offset);
      }
    }
  }

  // Build arrays of incoming and outcoming neighborhood ranks
  std::set<std::int32_t> in_neighbor_set, out_neighbor_set;
  for (const common::IndexMap& map : maps)
  {
    MPI_Comm neighbor_comm = map.comm(IndexMap::Direction::forward);
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
  MPI_Dist_graph_create_adjacent(maps.at(0).get().comm(), in_neighbors.size(),
                                 in_neighbors.data(), MPI_UNWEIGHTED,
                                 out_neighbors.size(), out_neighbors.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);

  // Figure out how much data to receive from each neighbor
  const int num_my_rows = indices.size();
  std::vector<int> num_rows_recv(indegree);
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
  std::vector<std::vector<int>> ghost_onwers_new(maps.size());
  for (std::size_t f = 0; f < maps.size(); ++f)
  {
    const int bs = maps[f].get().block_size();
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts
        = maps[f].get().ghosts();
    const Eigen::Array<int, Eigen::Dynamic, 1>& ghost_owners
        = maps[f].get().ghost_owner_rank();
    for (Eigen::Index i = 0; i < ghosts.rows(); ++i)
    {
      for (int j = 0; j < bs; ++j)
      {
        auto it = ghost_maps[f].find(bs * ghosts[i] + j);
        assert(it != ghost_maps[f].end());
        ghosts_new[f].push_back(it->second);
        ghost_onwers_new[f].push_back(ghost_owners[i]);
      }
    }
  }

  return {process_offset, std::move(local_offset), std::move(ghosts_new),
          std::move(ghost_onwers_new)};
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
                   const std::vector<std::int64_t>& ghosts,
                   const std::vector<int>& ghost_src_rank, int block_size)
    : IndexMap(mpi_comm, local_size,
               Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>(
                   ghosts.data(), ghosts.size()),
               ghost_src_rank, block_size)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(
    MPI_Comm mpi_comm, std::int32_t local_size,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
        ghosts,
    const std::vector<int>& ghost_src_rank, int block_size)
    : _block_size(block_size), _comm_owner_to_ghost(MPI_COMM_NULL),
      _comm_ghost_to_owner(MPI_COMM_NULL), _comm_symmetric(MPI_COMM_NULL),
      _ghosts(ghosts)
{
  assert(size_t(ghosts.size()) == ghost_src_rank.size());
  assert(ghost_src_rank == get_ghost_ranks(mpi_comm, local_size, _ghosts));

  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = (std::int64_t)local_size;
  MPI_Request request_scan;
  MPI_Iexscan(&local_size_tmp, &offset, 1, MPI_INT64_T, MPI_SUM, mpi_comm,
              &request_scan);

  // Send local size to sum reduction to get global size
  MPI_Request request;
  MPI_Iallreduce(&local_size_tmp, &_size_global, 1, MPI_INT64_T, MPI_SUM,
                 mpi_comm, &request);

  // TODO: move this call outside of the constructor
  // Use (i) the (remote) owner ranks for ghosts on this rank to (ii)
  // compute ranks that hold ghosts that this rank owns
  const std::set<int> owner_ranks_set(ghost_src_rank.begin(),
                                      ghost_src_rank.end());
  std::vector<std::int32_t> halo_dest_ranks
      = dolfinx::MPI::compute_source_ranks(mpi_comm, owner_ranks_set);
  std::sort(halo_dest_ranks.begin(), halo_dest_ranks.end());
  std::vector<std::int32_t> halo_src_ranks
      = std::vector<int>(owner_ranks_set.begin(), owner_ranks_set.end());

  // Map ghost owner rank to rank on neighborhood communicator
  int myrank = -1;
  MPI_Comm_rank(mpi_comm, &myrank);
  _ghost_owners.resize(ghosts.size());
  for (int j = 0; j < _ghosts.size(); ++j)
  {
    // Get rank of owner on the neighborhood communicator (rank of out
    // edge on _comm_owner_to_ghost)
    const auto it = std::find(halo_src_ranks.begin(), halo_src_ranks.end(),
                              ghost_src_rank[j]);
    assert(it != halo_src_ranks.end());
    const int p_neighbour = std::distance(halo_src_ranks.begin(), it);
    if (ghost_src_rank[j] == myrank)
    {
      throw std::runtime_error("IndexMap Error: Ghost in local range. Rank = "
                               + std::to_string(myrank)
                               + ", ghost = " + std::to_string(ghosts[j]));
    }

    // Store owner neighborhood rank for each ghost
    _ghost_owners[j] = p_neighbour;
  }

  // Create communicators with directional edges:
  // (0) owner -> ghost, (1) ghost -> owner, (2) two-way
  std::array<MPI_Comm, 3> comm_array = compute_asymmetric_communicators(
      mpi_comm, halo_src_ranks, halo_dest_ranks);
  _comm_owner_to_ghost = dolfinx::MPI::Comm(comm_array[0], false);
  _comm_ghost_to_owner = dolfinx::MPI::Comm(comm_array[1], false);
  _comm_symmetric = dolfinx::MPI::Comm(comm_array[2], false);

  // Compute owned forward (forward) indices
  auto [fwd_ind, fwd_sizes] = compute_forward_indices(
      _comm_ghost_to_owner.comm(), _ghosts, _ghost_owners);

  // Wait for MPI_Iexscan to complete (get offset)
  MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  _local_range = {offset, offset + local_size};

  _forward_indices.resize(fwd_ind.size());
  for (std::size_t i = 0; i < _forward_indices.size(); ++i)
    _forward_indices[i] = fwd_ind[i] - offset;

  _forward_sizes = std::move(fwd_sizes);

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMap::local_range() const
{
  return _local_range;
}
//-----------------------------------------------------------------------------
int IndexMap::block_size() const { return _block_size; }
//-----------------------------------------------------------------------------
std::int32_t IndexMap::num_ghosts() const { return _ghosts.rows(); }
//-----------------------------------------------------------------------------
std::int32_t IndexMap::size_local() const
{
  return _local_range[1] - _local_range[0];
}
//-----------------------------------------------------------------------------
std::int64_t IndexMap::size_global() const { return _size_global; }
//-----------------------------------------------------------------------------
const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& IndexMap::ghosts() const
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int64_t, Eigen::Dynamic, 1> IndexMap::local_to_global(
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&
        indices,
    bool blocked) const
{
  const std::int64_t global_offset = _local_range[0];
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> global(indices.rows());
  local_to_global_impl(global, indices, global_offset, local_size, _block_size,
                       _ghosts, blocked);

  return global;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
IndexMap::local_to_global(const std::vector<std::int32_t>& indices,
                          bool blocked) const
{
  const std::int64_t global_offset = _local_range[0];
  const std::int32_t local_size = _local_range[1] - _local_range[0];

  std::vector<std::int64_t> global(indices.size());
  Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> _global(
      global.data(), global.size());
  const Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
      _indices(indices.data(), indices.size());
  local_to_global_impl(_global, _indices, global_offset, local_size,
                       _block_size, _ghosts, blocked);
  return global;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> IndexMap::global_indices(bool blocked) const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  const std::int32_t num_ghosts = _ghosts.rows();
  const std::int64_t global_offset = _local_range[0];
  const int bs = blocked ? 1 : _block_size;

  std::vector<std::int64_t> global(bs * (local_size + num_ghosts));
  std::iota(global.begin(), global.begin() + bs * local_size,
            bs * global_offset);
  for (Eigen::Index i = 0; i < _ghosts.rows(); ++i)
  {
    for (int j = 0; j < bs; ++j)
      global[bs * (local_size + i) + j] = bs * _ghosts[i] + j;
  }

  return global;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
IndexMap::global_to_local(const std::vector<std::int64_t>& indices,
                          bool blocked) const
{
  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
      _indices(indices.data(), indices.size());
  return this->global_to_local(_indices, blocked);
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> IndexMap::global_to_local(
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
        indices,
    bool blocked) const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];

  std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts;
  for (Eigen::Index i = 0; i < _ghosts.rows(); ++i)
    global_local_ghosts.emplace_back(_ghosts[i], i + local_size);
  std::map<std::int64_t, std::int32_t> global_to_local(
      global_local_ghosts.begin(), global_local_ghosts.end());

  const int bs = blocked ? 1 : _block_size;
  std::vector<std::int32_t> local;
  const std::array<std::int64_t, 2> range = this->local_range();
  for (Eigen::Index i = 0; i < indices.size(); ++i)
  {
    const std::int64_t index = indices[i];
    if (index >= bs * range[0] and index < bs * range[1])
      local.push_back(index - bs * range[0]);
    else
    {
      const std::int64_t index_block = index / bs;
      if (auto it = global_to_local.find(index_block);
          it != global_to_local.end())
      {
        local.push_back(it->second * bs + index % bs);
      }
      else
        local.push_back(-1);
    }
  }

  return local;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& IndexMap::forward_indices() const
{
  return _forward_indices;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> IndexMap::ghost_owner_rank() const
{
  // Get neighbor processes
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);

  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_owner_to_ghost.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> owners(_ghost_owners.size());
  for (int i = 0; i < owners.size(); ++i)
    owners[i] = neighbors_in[_ghost_owners[i]];

  return owners;
}
//----------------------------------------------------------------------------
Eigen::Array<std::int64_t, Eigen::Dynamic, 1>
IndexMap::indices(bool unroll_block) const
{
  const int bs = unroll_block ? this->_block_size : 1;
  const std::array<std::int64_t, 2> local_range = this->local_range();
  const std::int32_t size_local = this->size_local() * bs;

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> indx(size_local
                                                     + num_ghosts() * bs);
  std::iota(indx.data(), indx.data() + size_local, bs * local_range[0]);
  for (Eigen::Index i = 0; i < num_ghosts(); ++i)
    for (Eigen::Index j = 0; j < bs; ++j)
      indx[size_local + bs * i + j] = bs * _ghosts[i] + j;

  return indx;
}
//----------------------------------------------------------------------------
MPI_Comm IndexMap::comm(Direction dir) const
{
  switch (dir)
  {
  case Direction::reverse:
    return _comm_ghost_to_owner.comm();
  case Direction::forward:
    return _comm_owner_to_ghost.comm();
  case Direction::symmetric:
    return _comm_symmetric.comm();
  default:
    throw std::runtime_error("Unknown edge direction for communicator.");
  }
}
//----------------------------------------------------------------------------
std::map<int, std::set<int>> IndexMap::compute_shared_indices() const
{
  std::map<int, std::set<int>> shared_indices;

  // Get number of neighbours
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);

  // Get neighbor processes
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_owner_to_ghost.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  // Get sharing of all owned indices
  for (std::size_t i = 0, c = 0; i < _forward_sizes.size(); ++i)
  {
    int dest_rank = neighbors_out[i];
    for (int j = 0; j < _forward_sizes[i]; ++j)
    {
      int idx = _forward_indices[c];
      shared_indices[idx].insert(dest_rank);
      ++c;
    }
  }

  // Pack shared indices that are ghost in more than one rank
  // and send forward
  std::vector<std::int64_t> fwd_sharing_data;
  std::vector<int> fwd_sharing_offsets{0};
  for (std::size_t i = 0, c = 0; i < _forward_sizes.size(); ++i)
  {
    for (int j = 0; j < _forward_sizes[i]; ++j, ++c)
    {
      int idx = _forward_indices[c];
      if (shared_indices[idx].size() > 1)
      {
        fwd_sharing_data.push_back(idx + _local_range[0]);
        fwd_sharing_data.push_back(shared_indices[idx].size());
        fwd_sharing_data.insert(fwd_sharing_data.end(),
                                shared_indices[idx].begin(),
                                shared_indices[idx].end());
      }
    }
    fwd_sharing_offsets.push_back(fwd_sharing_data.size());
  }

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> recv_sharing_data
      = dolfinx::MPI::neighbor_all_to_all(_comm_owner_to_ghost.comm(),
                                          fwd_sharing_offsets, fwd_sharing_data)
            .array();

  // Add ghost indices and onwers to map
  for (int i = 0; i < _ghosts.size(); ++i)
  {
    const std::int32_t idx = size_local() + i;
    const int np = _ghost_owners[i];
    shared_indices[idx].insert(neighbors_in[np]);
  }

  int myrank = -1;
  MPI_Comm_rank(_comm_owner_to_ghost.comm(), &myrank);

  // Add ranks (outside neighborhood) that share ghosts
  for (int i = 0; i < recv_sharing_data.size();)
  {
    auto it = std::find(_ghosts.data(), _ghosts.data() + _ghosts.size(),
                        recv_sharing_data[i]);
    const int idx = std::distance(_ghosts.data(), it) + size_local();
    int set_size = recv_sharing_data[i + 1];
    int set_pos = i + 2;
    for (int j = 0; j < set_size; j++)
      if (recv_sharing_data[set_pos + j] != myrank)
        shared_indices[idx].insert(recv_sharing_data[set_pos + j]);
    i += set_size + 2;
  }

  return shared_indices;
}
//-----------------------------------------------------------------------------
void IndexMap::scatter_fwd(const std::vector<std::int64_t>& local_data,
                           std::vector<std::int64_t>& remote_data, int n) const
{
  scatter_fwd_impl(local_data, remote_data, n);
}
//-----------------------------------------------------------------------------
void IndexMap::scatter_fwd(const std::vector<std::int32_t>& local_data,
                           std::vector<std::int32_t>& remote_data, int n) const
{
  scatter_fwd_impl(local_data, remote_data, n);
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
IndexMap::scatter_fwd(const std::vector<std::int64_t>& local_data, int n) const
{
  std::vector<std::int64_t> remote_data;
  scatter_fwd_impl(local_data, remote_data, n);
  return remote_data;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
IndexMap::scatter_fwd(const std::vector<std::int32_t>& local_data, int n) const
{
  std::vector<std::int32_t> remote_data;
  scatter_fwd_impl(local_data, remote_data, n);
  return remote_data;
}
//-----------------------------------------------------------------------------
void IndexMap::scatter_rev(std::vector<std::int64_t>& local_data,
                           const std::vector<std::int64_t>& remote_data, int n,
                           IndexMap::Mode op) const
{
  scatter_rev_impl(local_data, remote_data, n, op);
}
//-----------------------------------------------------------------------------
void IndexMap::scatter_rev(std::vector<std::int32_t>& local_data,
                           const std::vector<std::int32_t>& remote_data, int n,
                           IndexMap::Mode op) const
{
  scatter_rev_impl(local_data, remote_data, n, op);
}
//-----------------------------------------------------------------------------
template <typename T>
void IndexMap::scatter_fwd_impl(const std::vector<T>& local_data,
                                std::vector<T>& remote_data, int n) const
{

  // Get number of neighbours
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);

  // Get neighbor processes
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_owner_to_ghost.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  const std::int32_t _size_local = size_local();
  assert((int)local_data.size() == n * _size_local);
  remote_data.resize(n * _ghosts.size());

  // Create displacement vectors
  std::vector<std::int32_t> sizes_recv(indegree, 0);
  for (std::int32_t i = 0; i < _ghosts.size(); ++i)
    sizes_recv[_ghost_owners[i]] += n;

  std::vector<std::int32_t> displs_send(outdegree + 1, 0);
  std::vector<std::int32_t> displs_recv(indegree + 1, 0);
  std::vector<std::int32_t> sizes_send(outdegree, 0);
  for (int i = 0; i < outdegree; ++i)
  {
    sizes_send[i] = _forward_sizes[i] * n;
    displs_send[i + 1] = displs_send[i] + sizes_send[i];
  }

  for (int i = 0; i < indegree; ++i)
    displs_recv[i + 1] = displs_recv[i] + sizes_recv[i];

  // Copy into sending buffer
  std::vector<T> data_to_send(displs_send.back());
  for (std::size_t i = 0; i < _forward_indices.size(); ++i)
  {
    const int index = _forward_indices[i];
    for (int j = 0; j < n; ++j)
      data_to_send[i * n + j] = local_data[index * n + j];
  }

  // Send/receive data
  std::vector<T> data_to_recv(displs_recv.back());
  MPI_Neighbor_alltoallv(
      data_to_send.data(), sizes_send.data(), displs_send.data(),
      MPI::mpi_type<T>(), data_to_recv.data(), sizes_recv.data(),
      displs_recv.data(), MPI::mpi_type<T>(), _comm_owner_to_ghost.comm());

  // Copy into ghost area ("remote_data")
  std::vector<std::int32_t> displs(displs_recv);
  for (int i = 0; i < _ghosts.size(); ++i)
  {
    const int np = _ghost_owners[i];
    for (int j = 0; j < n; ++j)
      remote_data[i * n + j] = data_to_recv[displs[np] + j];
    displs[np] += n;
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void IndexMap::scatter_rev_impl(std::vector<T>& local_data,
                                const std::vector<T>& remote_data, int n,
                                IndexMap::Mode op) const
{
  assert((std::int32_t)remote_data.size() == n * num_ghosts());
  local_data.resize(n * size_local(), 0);

  // Get number of neighbours
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_ghost_to_owner.comm(), &indegree,
                                 &outdegree, &weighted);

  // Get neighbor processes
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_ghost_to_owner.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  // Compute number of items to send to each process
  std::vector<std::int32_t> send_sizes(outdegree, 0);
  std::vector<std::int32_t> recv_sizes(indegree, 0);
  for (int i = 0; i < _ghosts.size(); ++i)
    send_sizes[_ghost_owners[i]] += n;

  // Create displacement vectors
  std::vector<std::int32_t> displs_send(outdegree + 1, 0);
  std::vector<std::int32_t> displs_recv(indegree + 1, 0);
  for (int i = 0; i < indegree; ++i)
  {
    recv_sizes[i] = _forward_sizes[i] * n;
    displs_recv[i + 1] = displs_recv[i] + recv_sizes[i];
  }

  for (int i = 0; i < outdegree; ++i)
    displs_send[i + 1] = displs_send[i] + send_sizes[i];

  // Fill sending data
  std::vector<T> send_data(displs_send.back());
  std::vector<std::int32_t> displs(displs_send);
  for (std::int32_t i = 0; i < _ghosts.size(); ++i)
  {
    const int np = _ghost_owners[i];
    for (std::int32_t j = 0; j < n; ++j)
      send_data[displs[np] + j] = remote_data[i * n + j];
    displs[np] += n;
  }

  // Send and receive data
  std::vector<T> recv_data(displs_recv.back());
  MPI_Neighbor_alltoallv(
      send_data.data(), send_sizes.data(), displs_send.data(),
      MPI::mpi_type<T>(), recv_data.data(), recv_sizes.data(),
      displs_recv.data(), MPI::mpi_type<T>(), _comm_ghost_to_owner.comm());

  // Copy or accumulate into "local_data"
  if (op == Mode::insert)
  {
    for (std::size_t i = 0; i < _forward_indices.size(); ++i)
    {
      const int index = _forward_indices[i];
      for (int j = 0; j < n; ++j)
        local_data[index * n + j] = recv_data[i * n + j];
    }
  }
  else if (op == Mode::add)
  {
    for (std::size_t i = 0; i < _forward_indices.size(); ++i)
    {
      const int index = _forward_indices[i];
      for (int j = 0; j < n; ++j)
        local_data[index * n + j] += recv_data[i * n + j];
    }
  }
}
//-----------------------------------------------------------------------------
