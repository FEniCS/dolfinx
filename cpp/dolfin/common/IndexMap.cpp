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
IndexMap::IndexMap(MPI_Comm mpi_comm, std::size_t local_size,
                   const std::vector<std::size_t>& ghosts,
                   std::size_t block_size)
    : _mpi_comm(mpi_comm), _myrank(MPI::rank(mpi_comm)), _ghosts(ghosts.size()),
      _block_size(block_size)
{
  // Calculate offsets
  MPI::all_gather(_mpi_comm, local_size, _all_ranges);

  const std::size_t mpi_size = dolfin::MPI::size(_mpi_comm);
  for (std::size_t i = 1; i < mpi_size; ++i)
    _all_ranges[i] += _all_ranges[i - 1];

  _all_ranges.insert(_all_ranges.begin(), 0);

  _ghost_owners.resize(ghosts.size());
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
  return {{(std::int64_t)_all_ranges[_myrank],
           (std::int64_t)_all_ranges[_myrank + 1]}};
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
int IndexMap::owner(std::size_t global_index) const
{
  return std::upper_bound(_all_ranges.begin(), _all_ranges.end(), global_index)
         - _all_ranges.begin() - 1;
}
//-----------------------------------------------------------------------------
const EigenArrayXi32& IndexMap::ghost_owners() const { return _ghost_owners; }
//----------------------------------------------------------------------------
// MPI_Comm IndexMap::mpi_comm() const { return _mpi_comm.comm(); }
MPI_Comm IndexMap::mpi_comm() const { return _mpi_comm; }
//----------------------------------------------------------------------------
void IndexMap::scatter(std::vector<double>& local_data)
{
  // local_data should be the size of owned + ghost
  // Make the owned data available for reading by remote processes
  // and then get the ghost values from other processes

  const std::size_t nlocal = size_local();
  assert(local_data.size() == nlocal + num_ghosts());

  MPI_Win win;
  MPI_Win_create(local_data.data(), sizeof(double) * nlocal, sizeof(double),
                 MPI_INFO_NULL, _mpi_comm, &win);
  MPI_Win_fence(0, win);

  for (int i = 0; i < num_ghosts(); ++i)
  {
    // Remote process
    int p = _ghost_owners[i];
    // Index on remote process
    int remote_data_offset = _ghosts[i] - _all_ranges[p];

    // Stack up requests
    MPI_Get(local_data.data() + nlocal + i, 1, MPI_DOUBLE, p,
            remote_data_offset, 1, MPI_DOUBLE, win);
  }

  // Sync
  MPI_Win_fence(0, win);
  MPI_Win_free(&win);
}
