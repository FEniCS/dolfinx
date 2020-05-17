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
std::vector<int> get_ghost_ranks(
    dolfinx::MPI::Comm mpi_comm, std::int32_t local_size,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
        ghosts)
{
  int mpi_size = -1;
  MPI_Comm_size(mpi_comm.comm(), &mpi_size);
  std::vector<std::int32_t> local_sizes(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                mpi_comm.comm());

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

  // Build array of neighbourhood ranks
  std::set<std::int32_t> neighbour_set;
  for (const common::IndexMap& map : maps)
  {
    const std::vector<std::int32_t>& n = map.neighbours();
    neighbour_set.insert(n.begin(), n.end());
  }
  const std::vector<int> neighbours(neighbour_set.begin(), neighbour_set.end());

  // Create neighbourhood communicator
  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(maps.at(0).get().mpi_comm(), neighbours.size(),
                                 neighbours.data(), MPI_UNWEIGHTED,
                                 neighbours.size(), neighbours.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  int num_neighbours(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &num_neighbours, &outdegree, &weighted);
  assert(num_neighbours == outdegree);

  // Figure out how much data to receive from each neighbour
  const int num_my_rows = indices.size();
  std::vector<int> num_rows_recv(num_neighbours);
  MPI_Neighbor_allgather(&num_my_rows, 1, MPI_INT, num_rows_recv.data(), 1,
                         MPI_INT, comm);

  // Compute displacements for data to receive
  std::vector<int> disp(num_neighbours + 1, 0);
  std::partial_sum(num_rows_recv.begin(), num_rows_recv.end(),
                   disp.begin() + 1);

  // Send data to neighbours, and receive data
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
        = maps[f].get().ghost_owners();
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
                   const std::vector<int>& ghost_ranks, int block_size)
    : IndexMap(mpi_comm, local_size,
               Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>(
                   ghosts.data(), ghosts.size()),
               ghost_ranks, block_size)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(
    MPI_Comm mpi_comm, std::int32_t local_size,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
        ghosts,
    const std::vector<int>& ghost_ranks, int block_size)
    : _block_size(block_size), _mpi_comm(mpi_comm),
      _myrank(MPI::rank(mpi_comm)), _ghosts(ghosts),
      _ghost_owners(ghosts.size())
{

  assert(size_t(ghosts.size()) == ghost_ranks.size());

  int mpi_size = -1;
  MPI_Comm_size(_mpi_comm.comm(), &mpi_size);

#ifdef DEBUG
  assert(ghost_ranks == get_ghost_ranks(_mpi_comm, local_size, _ghosts));
#endif

  std::vector<std::int32_t> num_edges_out_per_proc(mpi_size, 0);
  for (int i = 0; i < ghosts.size(); ++i)
  {
    auto p = ghost_ranks[i];
    if (p == _myrank)
    {
      throw std::runtime_error("IndexMap Error: Ghost in local range. Rank = "
                               + std::to_string(_myrank)
                               + ", ghost = " + std::to_string(ghosts[i]));
    }
    num_edges_out_per_proc[p] += 1;
  }

  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  std::int64_t size_local = (std::int64_t)local_size;
  MPI_Exscan(&size_local, &offset, 1, MPI_INT64_T, MPI_SUM, _mpi_comm.comm());
  _local_range = {offset, offset + local_size};

  // Each MPI process sends its local size to reduction
  MPI_Request request;
  MPI_Iallreduce(&size_local, &_size_global, 1, MPI_INT64_T, MPI_SUM,
                 _mpi_comm.comm(), &request);

  // Send number of outgoing edges (ghost -> owner) to target processes,
  // and receive number of incoming edges (ghost <- owner) from each
  // source process
  std::vector<std::int32_t> num_edges_in_per_proc(mpi_size);
  MPI_Alltoall(num_edges_out_per_proc.data(), 1, MPI_INT32_T,
               num_edges_in_per_proc.data(), 1, MPI_INT32_T, _mpi_comm.comm());

  // Store number of out- and in-edges, and ranks of neighbourhood
  // processes
  std::vector<std::int32_t> in_edges_num, out_edges_num;
  for (std::int32_t i = 0; i < mpi_size; ++i)
  {
    if (num_edges_out_per_proc[i] > 0 or num_edges_in_per_proc[i] > 0)
    {
      _neighbours.push_back(i);
      in_edges_num.push_back(num_edges_in_per_proc[i]);
      out_edges_num.push_back(num_edges_out_per_proc[i]);
    }
  }

  // Create neighbourhood communicator. No communication is needed to
  // build the graph with complete adjacency information
  MPI_Comm neighbour_comm;
  MPI_Dist_graph_create_adjacent(
      _mpi_comm.comm(), _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED,
      _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &neighbour_comm);

  // Size of neighbourhood
  const int num_neighbours = _neighbours.size();

  // Check for 'symmetry' of the graph
#if DEBUG
  {
    std::vector<int> sources(num_neighbours), dests(num_neighbours);
    MPI_Dist_graph_neighbors(neighbour_comm, num_neighbours, sources.data(),
                             MPI_UNWEIGHTED, num_neighbours, dests.data(),
                             MPI_UNWEIGHTED);
    assert(sources == dests);
    assert(sources == _neighbours);
  }
#endif

  // Create displacement vectors
  std::vector<std::int32_t> disp_out(num_neighbours + 1, 0),
      disp_in(num_neighbours + 1, 0);
  std::partial_sum(out_edges_num.begin(), out_edges_num.end(),
                   disp_out.begin() + 1);
  std::partial_sum(in_edges_num.begin(), in_edges_num.end(),
                   disp_in.begin() + 1);

  // Get rank on neighbourhood communicator for each ghost, and for each
  // ghost compute the local index on the owning process
  std::vector<std::int32_t> out_indices(disp_out.back());
  std::vector<std::int32_t> disp(disp_out);
  for (int j = 0; j < _ghosts.size(); ++j)
  {
    // Get rank of owner process rank on global communicator
    const int p = ghost_ranks[j];

    // Get rank of owner on neighbourhood communicator
    const auto it = std::find(_neighbours.begin(), _neighbours.end(), p);
    assert(it != _neighbours.end());
    const int np = std::distance(_neighbours.begin(), it);

    // Store owner neighbourhood rank for each ghost
    _ghost_owners[j] = np;

    // Local on owning process
    out_indices[disp[np]] = _ghosts[j];
    disp[np] += 1;
  }

  //  May have repeated shared indices with different processes
  std::vector<std::int32_t> indices_in(disp_in.back());
  MPI_Neighbor_alltoallv(
      out_indices.data(), out_edges_num.data(), disp_out.data(), MPI_INT,
      indices_in.data(), // out
      in_edges_num.data(), disp_in.data(), MPI_INT, neighbour_comm);

  _forward_indices = std::move(indices_in);
  for (auto& value : _forward_indices)
    value -= offset;
  _forward_sizes = std::move(in_edges_num);

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  MPI_Comm_free(&neighbour_comm);
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
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> IndexMap::ghost_owners() const
{
  MPI_Comm neighbour_comm;
  MPI_Dist_graph_create_adjacent(
      _mpi_comm.comm(), _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED,
      _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &neighbour_comm);

  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbour_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  std::vector<int> neighbours(indegree), neighbours1(indegree);

  MPI_Dist_graph_neighbors(neighbour_comm, indegree, neighbours.data(),
                           MPI_UNWEIGHTED, outdegree, neighbours1.data(),
                           MPI_UNWEIGHTED);

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> proc_owners(
      _ghost_owners.size());
  for (int i = 0; i < proc_owners.size(); ++i)
    proc_owners[i] = neighbours[_ghost_owners[i]];

  MPI_Comm_free(&neighbour_comm);

  return proc_owners;
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
  {
    for (Eigen::Index j = 0; j < bs; ++j)
      indx[size_local + bs * i + j] = bs * _ghosts[i] + j;
  }

  return indx;
}
//----------------------------------------------------------------------------
MPI_Comm IndexMap::mpi_comm() const { return _mpi_comm.comm(); }
//----------------------------------------------------------------------------
const std::vector<std::int32_t>& IndexMap::neighbours() const
{
  return _neighbours;
}
//----------------------------------------------------------------------------
std::map<int, std::set<int>> IndexMap::compute_shared_indices() const
{
  std::map<int, std::set<int>> shared_indices;

  MPI_Comm neighbour_comm;
  MPI_Dist_graph_create_adjacent(
      _mpi_comm.comm(), _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED,
      _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &neighbour_comm);

  // Get neighbour processes
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbour_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  std::vector<int> neighbours(indegree), neighbours1(indegree);
  MPI_Dist_graph_neighbors(neighbour_comm, indegree, neighbours.data(),
                           MPI_UNWEIGHTED, outdegree, neighbours1.data(),
                           MPI_UNWEIGHTED);

  assert(neighbours.size() == _forward_sizes.size());

  // Get sharing of all owned indices
  int c = 0;
  for (std::size_t i = 0; i < _forward_sizes.size(); ++i)
  {
    int p = neighbours[i];
    for (int j = 0; j < _forward_sizes[i]; ++j)
    {
      int idx = _forward_indices[c];
      shared_indices[idx].insert(p);
      ++c;
    }
  }

  // Send forward
  std::vector<std::int64_t> fwd_sharing_data;
  std::vector<int> fwd_sharing_offsets = {0};
  c = 0;
  for (std::size_t i = 0; i < _forward_sizes.size(); ++i)
  {
    for (int j = 0; j < _forward_sizes[i]; ++j)
    {
      int idx = _forward_indices[c];
      fwd_sharing_data.push_back(shared_indices[idx].size());
      fwd_sharing_data.insert(fwd_sharing_data.end(),
                              shared_indices[idx].begin(),
                              shared_indices[idx].end());
      ++c;
    }
    fwd_sharing_offsets.push_back(fwd_sharing_data.size());
  }

  graph::AdjacencyList<std::int64_t> sharing = MPI::neighbor_all_to_all(
      neighbour_comm, fwd_sharing_offsets, fwd_sharing_data);
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> recv_sharing_offsets
      = sharing.offsets();
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& recv_sharing_data
      = sharing.array();

  // FIXME: The below is confusing and the std::set<int> inside the loop
  // should be avoided

  // Unpack
  for (int i = 0; i < _ghosts.size(); ++i)
  {
    int idx = size_local() + i;
    const int np = _ghost_owners[i];
    int p = neighbours[np];
    int& rp = recv_sharing_offsets[np];
    int ns = recv_sharing_data[rp];
    ++rp;
    std::set<int> procs(recv_sharing_data.data() + rp,
                        recv_sharing_data.data() + rp + ns);
    rp += ns;
    procs.insert(p);
    procs.erase(_myrank);
    shared_indices.insert({idx, procs});
  }

  MPI_Comm_free(&neighbour_comm);

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
  MPI_Comm neighbour_comm;
  MPI_Dist_graph_create_adjacent(
      _mpi_comm.comm(), _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED,
      _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &neighbour_comm);

#ifdef DEBUG
  // Check size of neighbourhood
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbour_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  assert(indegree == (int)_forward_sizes.size());
#endif
  const int num_neighbours = _forward_sizes.size();

  const std::int32_t _size_local = size_local();
  assert((int)local_data.size() == n * _size_local);
  remote_data.resize(n * _ghosts.size());

  // Create displacement vectors
  std::vector<std::int32_t> sizes_recv(num_neighbours, 0);
  for (std::int32_t i = 0; i < _ghosts.size(); ++i)
    sizes_recv[_ghost_owners[i]] += n;

  std::vector<std::int32_t> displs_send(num_neighbours + 1, 0);
  std::vector<std::int32_t> displs_recv(num_neighbours + 1, 0);
  std::vector<std::int32_t> sizes_send(num_neighbours, 0);
  for (std::int32_t i = 0; i < num_neighbours; ++i)
  {
    sizes_send[i] = _forward_sizes[i] * n;
    displs_send[i + 1] = displs_send[i] + sizes_send[i];
    displs_recv[i + 1] = displs_recv[i] + sizes_recv[i];
  }

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
      displs_recv.data(), MPI::mpi_type<T>(), neighbour_comm);

  // Copy into ghost area ("remote_data")
  std::vector<std::int32_t> displs(displs_recv);
  for (int i = 0; i < _ghosts.size(); ++i)
  {
    const int np = _ghost_owners[i];
    for (int j = 0; j < n; ++j)
      remote_data[i * n + j] = data_to_recv[displs[np] + j];
    displs[np] += n;
  }

  MPI_Comm_free(&neighbour_comm);
}
//-----------------------------------------------------------------------------
template <typename T>
void IndexMap::scatter_rev_impl(std::vector<T>& local_data,
                                const std::vector<T>& remote_data, int n,
                                IndexMap::Mode op) const
{
  assert((std::int32_t)remote_data.size() == n * num_ghosts());
  local_data.resize(n * size_local(), 0);

  MPI_Comm neighbour_comm;
  MPI_Dist_graph_create_adjacent(
      _mpi_comm.comm(), _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED,
      _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &neighbour_comm);

#ifdef DEBUG
  // Check size of neighbourhood
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbour_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  assert(indegree == (int)_forward_sizes.size());
#endif
  const int num_neighbours = _forward_sizes.size();

  // Compute number of items to send to each process
  std::vector<std::int32_t> send_sizes(num_neighbours, 0);
  for (std::int32_t i = 0; i < _ghosts.size(); ++i)
    send_sizes[_ghost_owners[i]] += n;

  // Create displacement vectors
  std::vector<std::int32_t> displs_send(num_neighbours + 1, 0);
  std::vector<std::int32_t> displs_recv(num_neighbours + 1, 0);
  std::vector<std::int32_t> recv_sizes(num_neighbours, 0);
  for (std::int32_t i = 0; i < num_neighbours; ++i)
  {
    recv_sizes[i] = _forward_sizes[i] * n;
    displs_send[i + 1] = displs_send[i] + send_sizes[i];
    displs_recv[i + 1] = displs_recv[i] + recv_sizes[i];
  }

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
      displs_recv.data(), MPI::mpi_type<T>(), neighbour_comm);

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

  MPI_Comm_free(&neighbour_comm);
}
//-----------------------------------------------------------------------------
