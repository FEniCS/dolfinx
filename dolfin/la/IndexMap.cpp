// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <algorithm>
#include <limits>
#include "IndexMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
IndexMap::IndexMap() : _block_size(1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm mpi_comm) : _mpi_comm(mpi_comm), _block_size(1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm mpi_comm, std::size_t local_size,
                   std::size_t block_size)
  : _mpi_comm(mpi_comm)
{
  init(local_size, block_size);
}
//-----------------------------------------------------------------------------
IndexMap::~IndexMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void IndexMap::init(std::size_t local_size, std::size_t block_size)
{
  _block_size = block_size;

  // Calculate offsets
  MPI::all_gather(_mpi_comm, local_size, _all_ranges);

  const std::size_t mpi_size = MPI::size(_mpi_comm);
  for (std::size_t i = 1; i != mpi_size; ++i)
    _all_ranges[i] += _all_ranges[i - 1];
  _all_ranges.insert(_all_ranges.begin(), 0);
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> IndexMap::local_range() const
{
  // Get my MPI rank
  const std::size_t rank = MPI::rank(_mpi_comm);

  if(_all_ranges.size() == 0)
  {
    warning("Asking for size of uninitialised range");
    return std::pair<std::size_t, std::size_t>(0, 0);
  }
  else
  {
    return std::make_pair(_block_size*_all_ranges[rank],
                          _block_size*_all_ranges[rank + 1]);
  }
}
//-----------------------------------------------------------------------------
std::size_t IndexMap::size(const IndexMap::MapSize type) const
{
  // Get my MPI rank
  const std::size_t rank = MPI::rank(_mpi_comm);

  if(_all_ranges.size() == 0)
  {
    warning("Asking for size of uninitialised range");
    return 0;
  }

  const std::size_t owned_size = _block_size*(_all_ranges[rank + 1]
                                              - _all_ranges[rank]);
  if (type == IndexMap::MapSize::OWNED)
    return owned_size;
  else if (type == IndexMap::MapSize::GLOBAL)
    return _all_ranges.back() * _block_size;

  const std::size_t unowned_size = _local_to_global.size()*_block_size;
  if (type == IndexMap::MapSize::ALL)
    return (owned_size + unowned_size);
  else if (type == IndexMap::MapSize::UNOWNED)
    return unowned_size;
  else
  {
    dolfin_error("IndexMap.cpp",
                 "get size",
                 "Unrecognised option for IndexMap::MapSize");
  }
  return 0;
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& IndexMap::local_to_global_unowned() const
{
  return _local_to_global;
}
//----------------------------------------------------------------------------
std::size_t IndexMap::local_to_global(std::size_t i) const
{
  const std::size_t local_size = size(IndexMap::MapSize::OWNED);
  const std::size_t global_offset = local_range().first;

  if (i < local_size)
    return (i + global_offset);
  else
  {
    const std::div_t div = std::div((i - local_size), _block_size);
    const int component = div.rem;
    const int index = div.quot;
    dolfin_assert((std::size_t) index < _local_to_global.size());
    return _block_size*_local_to_global[index] + component;
  }
}
//-----------------------------------------------------------------------------
void IndexMap::set_local_to_global(const std::vector<std::size_t>& indices)
{
  _local_to_global = indices;

  const std::size_t mpi_rank = MPI::rank(_mpi_comm);
  for (const auto &node : _local_to_global)
  {
    const std::size_t p
      = std::upper_bound(_all_ranges.begin(), _all_ranges.end(), node)
      - _all_ranges.begin() - 1;

    dolfin_assert(p != mpi_rank);
    _off_process_owner.push_back(p);
  }
}
//----------------------------------------------------------------------------
const std::vector<int>& IndexMap::off_process_owner() const
{
  return _off_process_owner;
}
//----------------------------------------------------------------------------
int IndexMap::block_size() const
{
  return _block_size;
}
//----------------------------------------------------------------------------
MPI_Comm IndexMap::mpi_comm() const
{
  return _mpi_comm;
}
//----------------------------------------------------------------------------
