// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include <algorithm>
#include <numeric>

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------------
IndexMap::IndexMap(
    MPI_Comm mpi_comm, std::int32_t local_size,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
        ghosts,
    int block_size)
    : block_size(block_size), _mpi_comm(mpi_comm), _myrank(MPI::rank(mpi_comm)),
      _ghosts(ghosts.size()), _ghost_owners(ghosts.size())
{
  // Copy ghosts vector
  for (int i = 0; i < ghosts.size(); ++i)
    _ghosts[i] = ghosts[i];

  // Calculate offsets
  int mpi_size = -1;
  MPI_Comm_size(mpi_comm, &mpi_size);
  std::vector<std::int32_t> local_sizes(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                mpi_comm);
  _all_ranges = std::vector<std::int64_t>(mpi_size + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   _all_ranges.begin() + 1);

  // Compute number of outgoing edges (ghost -> owner) to each remote
  // processes
  std::vector<int> ghost_owner_global(ghosts.size(), -1);
  std::vector<std::int32_t> num_edges_out_per_proc(mpi_size, 0);
  for (int i = 0; i < ghosts.size(); ++i)
  {
    const int p = owner(ghosts[i]);
    ghost_owner_global[i] = p;
    assert(ghost_owner_global[i] != _myrank);
    num_edges_out_per_proc[p] += 1;
  }

  // Send number of outgoing edges (ghost -> owner) to target processes,
  // and receive number of incoming edges (ghost <- owner) from each
  // source process
  std::vector<std::int32_t> num_edges_in_per_proc(mpi_size);
  MPI_Alltoall(num_edges_out_per_proc.data(), 1, MPI_INT32_T,
               num_edges_in_per_proc.data(), 1, MPI_INT32_T, _mpi_comm);

  // Store number of out- and in-edges, and ranks of neighbourhood
  // processes
  std::vector<std::int32_t> neighbours, in_edges_num, out_edges_num;
  for (std::int32_t i = 0; i < mpi_size; ++i)
  {
    if (num_edges_out_per_proc[i] > 0 or num_edges_in_per_proc[i] > 0)
    {
      neighbours.push_back(i);
      in_edges_num.push_back(num_edges_in_per_proc[i]);
      out_edges_num.push_back(num_edges_out_per_proc[i]);
    }
  }

  // Create neighbourhood communicator. No communication is needed to
  // build the graph with complete adjacency information
  MPI_Dist_graph_create_adjacent(
      _mpi_comm, neighbours.size(), neighbours.data(), MPI_UNWEIGHTED,
      neighbours.size(), neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &_neighbour_comm);

  // Size of neighbourhood
  const int num_neighbours = neighbours.size();

  // Check for 'symmetry' of the graph
#if DEBUG
  {
    std::vector<int> sources(num_neighbours), dests(num_neighbours);
    MPI_Dist_graph_neighbors(_neighbour_comm, num_neighbours, sources.data(),
                             NULL, num_neighbours, dests.data(), NULL);
    assert(sources == dests);
    assert(sources == neighbours);
  }
#endif

  // Create displacement vectors
  std::vector<int> disp_out(num_neighbours + 1, 0),
      disp_in(num_neighbours + 1, 0);
  std::partial_sum(out_edges_num.begin(), out_edges_num.end(),
                   disp_out.begin() + 1);
  std::partial_sum(in_edges_num.begin(), in_edges_num.end(),
                   disp_in.begin() + 1);

  // Get rank on neighbourhood communicator for each ghost, and for each
  // ghost compute the local index on the owning process
  std::vector<int> out_indices(disp_out.back());
  std::vector<int> disp(disp_out);
  for (int j = 0; j < _ghosts.size(); ++j)
  {
    // Get rank of owner process rank on global communicator
    const int p = ghost_owner_global[j];

    // Get rank of owner on neighbourhood communicator
    const auto it = std::find(neighbours.begin(), neighbours.end(), p);
    assert(it != neighbours.end());
    const int np = std::distance(neighbours.begin(), it);

    // Store owner neighbourhood rank for each ghost
    _ghost_owners[j] = np;

    // Local on owning process
    out_indices[disp[np]] = _ghosts[j] - _all_ranges[p];
    disp[np] += 1;
  }

  //  May have repeated shared indices with different processes
  std::vector<int> indices_in(disp_in.back());
  MPI_Neighbor_alltoallv(
      out_indices.data(), out_edges_num.data(), disp_out.data(), MPI_INT,
      indices_in.data(), // out
      in_edges_num.data(), disp_in.data(), MPI_INT, _neighbour_comm);

  _forward_indices = std::move(indices_in);
  _forward_sizes = std::move(in_edges_num);
}
//-----------------------------------------------------------------------------
IndexMap::~IndexMap() { MPI_Comm_free(&_neighbour_comm); }
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
std::map<std::int32_t, std::set<int>>
IndexMap::compute_forward_processes() const
{
  // Get neighbour processes
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_neighbour_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  std::vector<int> neighbours(indegree), neighbours1(indegree),
      weights(indegree), weights1(indegree);

  MPI_Dist_graph_neighbors(_neighbour_comm, indegree, neighbours.data(),
                           weights.data(), outdegree, neighbours1.data(),
                           weights1.data());

  assert(neighbours.size() == _forward_sizes.size());

  std::map<std::int32_t, std::set<int>> sh_map;
  int k = 0;
  // Iterate through each neighbour (i)
  for (std::size_t i = 0; i < _forward_sizes.size(); ++i)
  {
    for (int j = 0; j < _forward_sizes[i]; ++j)
    {
      // Add map entry from this index to the process it is shared with
      sh_map[_forward_indices[k]].insert(neighbours[i]);
      ++k;
    }
  }
  return sh_map;
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
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_neighbour_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  std::vector<int> neighbours(indegree), neighbours1(indegree),
      weights(indegree), weights1(indegree);

  MPI_Dist_graph_neighbors(_neighbour_comm, indegree, neighbours.data(),
                           weights.data(), outdegree, neighbours1.data(),
                           weights1.data());

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> proc_owners(
      _ghost_owners.size());
  for (int i = 0; i < proc_owners.size(); ++i)
    proc_owners[i] = neighbours[_ghost_owners[i]];

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
MPI_Comm IndexMap::mpi_comm_neighborhood() const { return _neighbour_comm; }
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

#ifdef DEBUG
  // Check size of neighbourhood
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_neighbour_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  assert(indegree == (int)_forward_sizes.size());
#endif
  const int num_neighbours = _forward_sizes.size();

  const std::int32_t _size_local = size_local();
  assert((int)local_data.size() == n * _size_local);
  remote_data.resize(n * _ghosts.rows());

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

  // Send/receive daat
  std::vector<T> data_to_recv(displs_recv.back());
  MPI_Neighbor_alltoallv(
      data_to_send.data(), sizes_send.data(), displs_send.data(),
      MPI::mpi_type<T>(), data_to_recv.data(), sizes_recv.data(),
      displs_recv.data(), MPI::mpi_type<T>(), _neighbour_comm);

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

#ifdef DEBUG
  // Check size of neighbourhood
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_neighbour_comm, &indegree, &outdegree,
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
      displs_recv.data(), MPI::mpi_type<T>(), _neighbour_comm);

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
