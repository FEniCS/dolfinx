// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include <algorithm>
#include <functional>
#include <map>
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
  MPI::all_gather(_mpi_comm, (std::int64_t)local_size, _all_ranges);

  const std::int32_t mpi_size = dolfin::MPI::size(_mpi_comm);
  for (std::int32_t i = 1; i < mpi_size; ++i)
    _all_ranges[i] += _all_ranges[i - 1];

  _all_ranges.insert(_all_ranges.begin(), 0);

  for (std::size_t i = 0; i < ghosts.size(); ++i)
  {
    _ghosts[i] = ghosts[i];
    _ghost_owners[i] = owner(ghosts[i]);
    assert(_ghost_owners[i] != _myrank);
  }

  std::set<std::int32_t> ghost_set(_ghost_owners.data(),
                                   _ghost_owners.data() + _ghost_owners.size());
  std::vector<std::int32_t> sources(1, _myrank);
  std::vector<std::int32_t> degrees(1, ghost_set.size());
  std::vector<std::int32_t> dests(ghost_set.begin(), ghost_set.end());

  // TODO: Avoid creating the graph twice
  MPI_Dist_graph_create(_mpi_comm, sources.size(), sources.data(),
                        degrees.data(), dests.data(), MPI_UNWEIGHTED,
                        MPI_INFO_NULL, false, &_neighbour_comm);

  std::int32_t in_degree, out_degree, w;
  MPI_Dist_graph_neighbors_count(_neighbour_comm, &in_degree, &out_degree, &w);

  sources.resize(in_degree);
  dests.resize(out_degree);
  MPI_Dist_graph_neighbors(_neighbour_comm, in_degree, sources.data(), NULL,
                           out_degree, dests.data(), NULL);

  // Gather information for complete two-sided communication
  std::sort(sources.begin(), sources.end());
  std::sort(dests.begin(), dests.end());
  std::set_union(sources.begin(), sources.end(), dests.begin(), dests.end(),
                 std::back_inserter(_neighbours));
  std::int32_t num_neighbours = _neighbours.size();
  // No communication is needed to build the graph with complete adjacent
  // information
  MPI_Dist_graph_create_adjacent(
      _mpi_comm, _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED,
      _neighbours.size(), _neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &_neighbour_comm);

  sources.resize(num_neighbours);
  dests.resize(num_neighbours);
  MPI_Dist_graph_neighbors(_neighbour_comm, num_neighbours, sources.data(),
                           NULL, num_neighbours, dests.data(), NULL);
  assert(sources == dests);

  // Number of indices to send and receive
  _reverse_sizes.resize(num_neighbours);
  _forward_sizes.resize(num_neighbours);
  for (std::int32_t i = 0; i < num_neighbours; ++i)
  {
    int count = std::count(_ghost_owners.data(),
                           _ghost_owners.data() + _ghost_owners.size(),
                           _neighbours[i]);
    _reverse_sizes[i] = count;
  }

  MPI_Neighbor_alltoall(_reverse_sizes.data(), 1, MPI_INT,
                        _forward_sizes.data(), 1, MPI_INT, _neighbour_comm);

  // Create vectors for forward communication
  int n_send = std::accumulate(_reverse_sizes.begin(), _reverse_sizes.end(), 0);
  std::vector<std::int32_t> reverse_indices(n_send);
  int n_recv = std::accumulate(_forward_sizes.begin(), _forward_sizes.end(), 0);
  _forward_indices.resize(n_recv);

  // Create displacement vector
  std::vector<std::int32_t> displs_reverse(num_neighbours, 0);
  std::vector<std::int32_t> displs_forward(num_neighbours, 0);
  for (std::int32_t i = 1; i < num_neighbours; ++i)
  {
    displs_reverse[i] = displs_reverse[i - 1] + _reverse_sizes[i - 1];
    displs_forward[i] = displs_forward[i - 1] + _forward_sizes[i - 1];
  }

  std::vector<std::int32_t> displs(displs_reverse);
  int nb_ind;
  std::vector<std::int32_t>::iterator it;
  for (std::int32_t j = 0; j < _ghosts.size(); ++j)
  {
    it = std::find(_neighbours.begin(), _neighbours.end(), _ghost_owners[j]);
    nb_ind = it - _neighbours.begin();
    reverse_indices[displs[nb_ind]]
        = _ghosts[j] - _all_ranges[_neighbours[nb_ind]];
    displs[nb_ind] += 1;
  }

  //  May have repeated shared indidices with different processes
  MPI_Neighbor_alltoallv(reverse_indices.data(), _reverse_sizes.data(),
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
const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
IndexMap::ghost_owners() const
{
  return _ghost_owners;
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
                           MPI_Op op) const
{
  scatter_rev_impl(local_data, remote_data, n, op);
}
//-----------------------------------------------------------------------------
void IndexMap::scatter_rev(std::vector<std::int32_t>& local_data,
                           const std::vector<std::int32_t>& remote_data, int n,
                           MPI_Op op) const
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
  std::vector<std::int32_t> displs_send(num_neighbours, 0);
  std::vector<std::int32_t> displs_recv(num_neighbours, 0);
  std::vector<std::int32_t> send_sizes(num_neighbours, 0);
  std::vector<std::int32_t> recv_sizes(num_neighbours, 0);
  for (std::int32_t i = 0; i < num_neighbours; ++i)
  {
    send_sizes[i] = _forward_sizes[i] * n;
    recv_sizes[i] = _reverse_sizes[i] * n;
    if (i > 1)
    {
      displs_send[i] = displs_send[i - 1] + send_sizes[i - 1];
      displs_recv[i] = displs_recv[i - 1] + send_sizes[i - 1];
    }
  }

  // Allocate buffers for sending an receiving data
  int n_send = std::accumulate(send_sizes.begin(), send_sizes.end(), 0);
  int n_recv = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
  std::vector<T> data_to_send(n_send);
  std::vector<T> data_to_recv(n_recv);
  int nb_ind;
  for (std::size_t i = 0; i < _forward_indices.size(); ++i)
  {
    int index = _forward_indices[i];
    for (std::int32_t j = 0; j < n; ++j)
      data_to_send[i * n + j] = local_data[n * index + j];
  }

  MPI_Neighbor_alltoallv(
      data_to_send.data(), send_sizes.data(), displs_send.data(),
      MPI::mpi_type<T>(), data_to_recv.data(), recv_sizes.data(),
      displs_recv.data(), MPI::mpi_type<T>(), _neighbour_comm);

  std::vector<std::int32_t> offsets_recv(displs_recv);
  for (int i = 0; i < _ghosts.size(); ++i)
  {
    nb_ind = std::find(_neighbours.begin(), _neighbours.end(), _ghost_owners[i])
             - _neighbours.begin();
    for (std::int32_t j = 0; j < n; ++j)
      remote_data[i * n + j] = data_to_recv[offsets_recv[nb_ind] + j];
    offsets_recv[nb_ind] += n;
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void IndexMap::scatter_rev_impl(std::vector<T>& local_data,
                                const std::vector<T>& remote_data, int n,
                                MPI_Op op) const
{
  assert((std::int32_t)remote_data.size() == n * num_ghosts());
  local_data.resize(n * size_local(), 0);

  // Open window into local data array
  MPI_Win win;
  MPI_Win_create(local_data.data(), sizeof(T) * n * size_local(), sizeof(T),
                 MPI_INFO_NULL, _mpi_comm, &win);
  MPI_Win_fence(0, win);

  // 'Put' (accumulate) ghost data onto owning process
  for (int i = 0; i < num_ghosts(); ++i)
  {

    // Remote owning process
    const int p = _ghost_owners[i];

    // Index on remote process
    const int remote_data_offset = _ghosts[i] - _all_ranges[p];

    // Stack up requests (sum)
    MPI_Accumulate(remote_data.data() + n * i, n, MPI::mpi_type<T>(), p,
                   n * remote_data_offset, n, MPI::mpi_type<T>(), op, win);
  }

  // Synchronise and free window
  MPI_Win_fence(0, win);
  MPI_Win_free(&win);
}
//-----------------------------------------------------------------------------
