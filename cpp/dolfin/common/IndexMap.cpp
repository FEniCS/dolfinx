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
                   std::size_t block_size)
    : _mpi_comm(mpi_comm), _rank(MPI::rank(mpi_comm)), _block_size(block_size)
{
  // Calculate offsets
  MPI::all_gather(_mpi_comm.comm(), local_size, _all_ranges);

  const std::size_t mpi_size = _mpi_comm.size();
  for (std::size_t i = 1; i != mpi_size; ++i)
    _all_ranges[i] += _all_ranges[i - 1];

  _all_ranges.insert(_all_ranges.begin(), 0);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMap::local_range() const
{
  if (_all_ranges.size() == 0)
  {
    log::warning("Asking for size of uninitialised range");
    return {{0, 0}};
  }
  else
    return {{(std::int64_t)_all_ranges[_rank],
             (std::int64_t)_all_ranges[_rank + 1]}};
}
//-----------------------------------------------------------------------------
std::size_t IndexMap::size(const IndexMap::MapSize type) const
{
  if (_all_ranges.size() == 0)
  {
    log::warning("Asking for size of uninitialised range");
    return 0;
  }

  const std::size_t owned_size = _all_ranges[_rank + 1] - _all_ranges[_rank];

  if (type == IndexMap::MapSize::OWNED)
    return owned_size;
  else if (type == IndexMap::MapSize::GLOBAL)
    return _all_ranges.back();

  const std::size_t unowned_size = _local_to_global.size();
  if (type == IndexMap::MapSize::ALL)
    return (owned_size + unowned_size);
  else if (type == IndexMap::MapSize::UNOWNED)
    return unowned_size;
  else
  {
    log::dolfin_error("IndexMap.cpp", "get size",
                      "Unrecognised option for IndexMap::MapSize");
  }

  return 0;
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& IndexMap::local_to_global_unowned() const
{
  return _local_to_global;
}
//-----------------------------------------------------------------------------
void IndexMap::set_block_local_to_global(
    const std::vector<std::size_t>& indices)
{
  _local_to_global = indices;

  for (const auto& node : _local_to_global)
  {
    const std::size_t p = global_block_index_owner(node);
    assert(p != _rank);
    _off_process_owner.push_back(p);
  }
}
//-----------------------------------------------------------------------------
int IndexMap::global_block_index_owner(std::size_t index) const
{
  const int p = std::upper_bound(_all_ranges.begin(), _all_ranges.end(), index)
                - _all_ranges.begin() - 1;
  return p;
}
//-----------------------------------------------------------------------------
const std::vector<int>& IndexMap::block_off_process_owner() const
{
  return _off_process_owner;
}
//-----------------------------------------------------------------------------
int IndexMap::block_size() const { return _block_size; }
//----------------------------------------------------------------------------
MPI_Comm IndexMap::mpi_comm() const { return _mpi_comm.comm(); }
//----------------------------------------------------------------------------
