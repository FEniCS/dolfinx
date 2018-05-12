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
  MPI::all_gather(_mpi_comm.comm(), local_size, _all_ranges);

  const std::size_t mpi_size = _mpi_comm.size();
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
std::size_t IndexMap::size(const IndexMap::MapSize type) const
{
  switch (type)
  {
  case IndexMap::MapSize::OWNED:
    return _all_ranges[_myrank + 1] - _all_ranges[_myrank];
  case IndexMap::MapSize::GLOBAL:
    return _all_ranges.back();
  case IndexMap::MapSize::ALL:
    return _all_ranges[_myrank + 1] - _all_ranges[_myrank] + _ghosts.size();
  case IndexMap::MapSize::GHOSTS:
    return _ghosts.size();
  default:
    throw std::runtime_error("Unknown size type");
    return 0;
  }
}
//-----------------------------------------------------------------------------
const EigenArrayXi64& IndexMap::ghosts() const { return _ghosts; }
//-----------------------------------------------------------------------------
int IndexMap::owner(std::size_t global_index) const
{
  return std::upper_bound(_all_ranges.begin(), _all_ranges.end(), global_index)
         - _all_ranges.begin() - 1;
}
//-----------------------------------------------------------------------------
const EigenArrayXi32& IndexMap::ghost_owners() const { return _ghost_owners; }
//----------------------------------------------------------------------------
MPI_Comm IndexMap::mpi_comm() const { return _mpi_comm.comm(); }
//----------------------------------------------------------------------------
