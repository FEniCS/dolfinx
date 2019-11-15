// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include <algorithm>
#include <set>
#include <vector>

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
                   const std::vector<std::int64_t>& ghosts,
                   std::size_t block_size)
    : block_size(block_size), _mpi_comm(mpi_comm), _myrank(MPI::rank(mpi_comm)),
      _ghosts(ghosts.size()), _ghost_owners(ghosts.size())
{
  // Calculate offsets
  int mpi_size = -1;
  MPI_Comm_size(mpi_comm, &mpi_size);
  std::vector<std::int32_t> local_sizes(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                mpi_comm);
  _all_ranges = std::vector<std::int64_t>(mpi_size + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   _all_ranges.begin() + 1);

  // Compute number of outgoing edges to other processes
  std::vector<std::int32_t> nghosts_send(mpi_size, 0);
  for (std::size_t i = 0; i < ghosts.size(); ++i)
  {
    const int p = owner(ghosts[i]);
    _ghosts[i] = ghosts[i];
    _ghost_owners[i] = p;
    assert(_ghost_owners[i] != _myrank);
    nghosts_send[p] += 1;
  }

  // Send number of outgoing edges (target) processes, and receive
  // process numbers of incoming edges from each process
  std::vector<std::int32_t> nghosts_recv(mpi_size);
  MPI_Alltoall(nghosts_send.data(), 1, MPI_INT32_T, nghosts_recv.data(), 1,
               MPI_INT32_T, _mpi_comm);

  // Compute number of out- and in-edges, and ranks of neighbourhood
  // processes
  std::vector<std::int32_t> out_edges_num;
  for (std::int32_t i = 0; i < mpi_size; ++i)
  {
    if (nghosts_send[i] > 0 or nghosts_recv[i] > 0)
    {
      _neighbours.push_back(i);
      _forward_sizes.push_back(nghosts_recv[i]);
      out_edges_num.push_back(nghosts_send[i]);
    }
  }

  std::int32_t num_neighbours = _neighbours.size();

  // No further communication is needed to build the graph with complete
  // adjacency information
  MPI_Dist_graph_create_adjacent(
      _mpi_comm, _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED,
      _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &_neighbour_comm);

  std::vector<std::int32_t> sources(num_neighbours);
  std::vector<std::int32_t> dests(num_neighbours);
  MPI_Dist_graph_neighbors(_neighbour_comm, num_neighbours, sources.data(),
                           NULL, num_neighbours, dests.data(), NULL);
  assert(sources == dests);
  assert(sources == _neighbours);

  // Create displacement vector
  std::vector<std::int32_t> displs_reverse(num_neighbours + 1, 0);
  std::vector<std::int32_t> displs_forward(num_neighbours + 1, 0);
  for (std::int32_t i = 0; i < num_neighbours; ++i)
  {
    displs_reverse[i + 1] = displs_reverse[i] + out_edges_num[i];
    displs_forward[i + 1] = displs_forward[i] + _forward_sizes[i];
  }

  // Create vectors for forward communication
  int n_send = displs_reverse.back();
  std::vector<std::int32_t> reverse_indices(n_send);
  int n_recv = displs_forward.back();
  _forward_indices.resize(n_recv);

  std::vector<std::int32_t> displs(displs_reverse);
  for (std::int32_t j = 0; j < _ghosts.size(); ++j)
  {
    const int p = _ghost_owners[j];
    const auto it = std::find(_neighbours.begin(), _neighbours.end(), p);
    assert(it != _neighbours.end());
    const int np = it - _neighbours.begin();

    // Convert to neighbour number instead of global process number
    _ghost_owners[j] = np;

    reverse_indices[displs[np]] = _ghosts[j] - _all_ranges[p];
    ++displs[np];
  }

  //  May have repeated shared indices with different processes
  MPI_Neighbor_alltoallv(reverse_indices.data(), out_edges_num.data(),
                         displs_reverse.data(), MPI_INT,
                         _forward_indices.data(), _forward_sizes.data(),
                         displs_forward.data(), MPI_INT, _neighbour_comm);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMap::local_range() const
{
  return {{_all_ranges[_myrank], _all_ranges[_myrank + 1]}};
}
//-----------------------------------------------------------------------------
std::int32_t IndexMap::num_ghosts() const { return _ghosts.size(); }
//-----------------------------------------------------------------------------
std::int32_t IndexMap::size_local() const
{
  return _all_ranges[_myrank + 1] - _all_ranges[_myrank];
}
//-----------------------------------------------------------------------------
std::int64_t IndexMap::size_global() const { return _all_ranges.back(); }
//-----------------------------------------------------------------------------
const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& IndexMap::ghosts() const
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
int IndexMap::owner(std::int64_t global_index) const
{
  auto it
      = std::upper_bound(_all_ranges.begin(), _all_ranges.end(), global_index);
  return std::distance(_all_ranges.begin(), it) - 1;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> IndexMap::ghost_owners() const
{
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> proc_owners(
      _ghost_owners.size());
  for (int i = 0; i < proc_owners.size(); ++i)
    proc_owners[i] = _neighbours[_ghost_owners[i]];

  return proc_owners;
}
//----------------------------------------------------------------------------
Eigen::Array<std::int64_t, Eigen::Dynamic, 1>
IndexMap::indices(bool unroll_block) const
{
  const int bs = unroll_block ? this->block_size : 1;
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
MPI_Comm IndexMap::mpi_comm() const { return _mpi_comm; }
//----------------------------------------------------------------------------
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
  const std::size_t _size_local = size_local();
  const int num_neighbours = _neighbours.size();
  assert(local_data.size() == n * _size_local);
  remote_data.resize(n * num_ghosts());

  // Create displacement vectors
  std::vector<std::int32_t> displs_send(num_neighbours + 1, 0);
  std::vector<std::int32_t> displs_recv(num_neighbours + 1, 0);
  std::vector<std::int32_t> send_sizes(num_neighbours, 0);
  std::vector<std::int32_t> recv_sizes(num_neighbours, 0);

  for (std::int32_t i = 0; i < _ghosts.size(); ++i)
    recv_sizes[_ghost_owners[i]] += n;

  for (std::int32_t i = 0; i < num_neighbours; ++i)
  {
    send_sizes[i] = _forward_sizes[i] * n;
    displs_send[i + 1] = displs_send[i] + send_sizes[i];
    displs_recv[i + 1] = displs_recv[i] + recv_sizes[i];
  }

  // Allocate buffers for sending and receiving data
  const int n_send = displs_send.back();
  std::vector<T> data_to_send(n_send);
  const int n_recv = displs_recv.back();
  std::vector<T> data_to_recv(n_recv);

  // Copy into sending buffer
  for (std::size_t i = 0; i < _forward_indices.size(); ++i)
  {
    const int index = _forward_indices[i];
    for (std::int32_t j = 0; j < n; ++j)
      data_to_send[i * n + j] = local_data[index * n + j];
  }

  MPI_Neighbor_alltoallv(
      data_to_send.data(), send_sizes.data(), displs_send.data(),
      MPI::mpi_type<T>(), data_to_recv.data(), recv_sizes.data(),
      displs_recv.data(), MPI::mpi_type<T>(), _neighbour_comm);

  // Copy into ghost area ("remote_data")
  std::vector<std::int32_t> displs(displs_recv);
  for (int i = 0; i < _ghosts.size(); ++i)
  {
    const int np = _ghost_owners[i];
    for (std::int32_t j = 0; j < n; ++j)
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

  int num_neighbours = _neighbours.size();

  // Create displacement vectors
  std::vector<std::int32_t> displs_send(num_neighbours + 1, 0);
  std::vector<std::int32_t> displs_recv(num_neighbours + 1, 0);
  std::vector<std::int32_t> send_sizes(num_neighbours, 0);
  std::vector<std::int32_t> recv_sizes(num_neighbours, 0);

  for (std::int32_t i = 0; i < _ghosts.size(); ++i)
    send_sizes[_ghost_owners[i]] += n;

  for (std::int32_t i = 0; i < num_neighbours; ++i)
  {
    recv_sizes[i] = _forward_sizes[i] * n;
    displs_send[i + 1] = displs_send[i] + send_sizes[i];
    displs_recv[i + 1] = displs_recv[i] + recv_sizes[i];
  }

  // Create vectors for data
  const int n_send = displs_send.back();
  std::vector<T> send_data(n_send);
  const int n_recv = displs_recv.back();
  std::vector<T> recv_data(n_recv);

  // Fill sending data
  std::vector<std::int32_t> displs(displs_send);
  for (std::int32_t i = 0; i < _ghosts.size(); ++i)
  {
    const int np = _ghost_owners[i];
    for (std::int32_t j = 0; j < n; ++j)
      send_data[displs[np] + j] = remote_data[i * n + j];
    displs[np] += n;
  }

  // May have repeated shared indices with different processes
  MPI_Neighbor_alltoallv(
      send_data.data(), send_sizes.data(), displs_send.data(),
      MPI::mpi_type<T>(), recv_data.data(), recv_sizes.data(),
      displs_recv.data(), MPI::mpi_type<T>(), _neighbour_comm);

  // Copy or accumulate into "local_data"
  if (op == Mode::insert)
  {
    for (std::size_t i = 0; i < _forward_indices.size(); ++i)
    {
      const int index = _forward_indices[i];
      for (std::int32_t j = 0; j < n; ++j)
        local_data[index * n + j] = recv_data[i * n + j];
    }
  }
  else if (op == Mode::add)
  {
    for (std::size_t i = 0; i < _forward_indices.size(); ++i)
    {
      const int index = _forward_indices[i];
      for (std::int32_t j = 0; j < n; ++j)
        local_data[index * n + j] += recv_data[i * n + j];
    }
  }
}
//-----------------------------------------------------------------------------
