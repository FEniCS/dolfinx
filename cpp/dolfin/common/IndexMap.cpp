// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include <algorithm>
#include <limits>

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
                   const std::vector<std::size_t>& ghosts,
                   std::size_t block_size)
    : _mpi_comm(mpi_comm), _myrank(MPI::rank(mpi_comm)), _ghosts(ghosts.size()),
      _ghost_owners(ghosts.size()), _block_size(block_size)
{
  // Calculate offsets
  MPI::all_gather(_mpi_comm, (std::int64_t)local_size, _all_ranges);

  const std::size_t mpi_size = dolfin::MPI::size(_mpi_comm);
  for (std::size_t i = 1; i < mpi_size; ++i)
    _all_ranges[i] += _all_ranges[i - 1];

  _all_ranges.insert(_all_ranges.begin(), 0);

  for (std::size_t i = 0; i < ghosts.size(); ++i)
  {
    _ghosts[i] = ghosts[i];
    _ghost_owners[i] = owner(ghosts[i]);
    assert(_ghost_owners[i] != _myrank);
  }
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
                   const std::vector<std::int64_t>& ghosts,
                   std::size_t block_size)
    : _mpi_comm(mpi_comm), _myrank(MPI::rank(mpi_comm)), _ghosts(ghosts.size()),
      _ghost_owners(ghosts.size()), _block_size(block_size)
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
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMap::local_range() const
{
  return {{_all_ranges[_myrank], _all_ranges[_myrank + 1]}};
}
//-----------------------------------------------------------------------------
int IndexMap::block_size() const { return _block_size; }
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

  // return std::upper_bound(_all_ranges.begin(), _all_ranges.end(),
  // global_index)
  //        - _all_ranges.begin() - 1;
}
//-----------------------------------------------------------------------------
const EigenArrayXi32& IndexMap::ghost_owners() const { return _ghost_owners; }
//----------------------------------------------------------------------------
// MPI_Comm IndexMap::mpi_comm() const { return _mpi_comm.comm(); }
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
void IndexMap::scatter_rev(std::vector<std::int64_t>& local_data,
                           const std::vector<std::int64_t>& remote_data,
                           int n, MPI_Op op) const
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
  assert(local_data.size() == n * _size_local);
  remote_data.resize(n * num_ghosts());

  // Open window into owned data
  MPI_Win win;
  MPI_Win_create(const_cast<T*>(local_data.data()), sizeof(T) * n * _size_local,
                 sizeof(T), MPI_INFO_NULL, _mpi_comm, &win);
  MPI_Win_fence(0, win);

  // Fetch ghost data from owner
  for (int i = 0; i < num_ghosts(); ++i)
  {
    // Remote process rank
    const int p = _ghost_owners[i];

    // Index on remote process
    const int remote_data_offset = _ghosts[i] - _all_ranges[p];

    // Stack up requests
    MPI_Get(remote_data.data() + n * i, n, dolfin::MPI::mpi_type<T>(), p,
            n * remote_data_offset, n, dolfin::MPI::mpi_type<T>(), win);
  }

  // Synchronise and free window
  MPI_Win_fence(0, win);
  MPI_Win_free(&win);
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

    if (remote_data[i] == 461)
    {
      if (i == 3)
        std::cout << "test value: " << remote_data[i] << ", " << p << ", "
                  << remote_data_offset << ", " << MPI::rank(MPI_COMM_WORLD)
                  << std::endl;
    }
    // if (MPI::rank(MPI_COMM_WORLD) == 0)
    // {
    //   if (i == 3)
    //     std::cout << "test value: " << remote_data[i] << ", " << p << ", "
    //               << remote_data_offset << std::endl;
    // }

    // Stack up requests (sum)
    MPI_Accumulate(remote_data.data() + n * i, n, MPI::mpi_type<T>(), p,
                   remote_data_offset, n, MPI::mpi_type<T>(), op, win);
  }

  // Synchronise and free window
  MPI_Win_fence(0, win);
  MPI_Win_free(&win);
}
//-----------------------------------------------------------------------------