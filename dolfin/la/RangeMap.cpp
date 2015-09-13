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
//

#include <algorithm>
#include "RangeMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void RangeMap::init(std::size_t local_size, std::size_t block_size)
{
  _block_size = block_size;

  // Calculate offsets
  MPI::all_gather(_mpi_comm, local_size, _all_ranges);

  const std::size_t mpi_rank = MPI::rank(_mpi_comm);
  const std::size_t mpi_size = MPI::size(_mpi_comm);
  for (std::size_t i = 1; i != mpi_size; ++i)
    _all_ranges[i] += _all_ranges[i - 1];
  _all_ranges.insert(_all_ranges.begin(), 0);

  //  _local_range.first = _all_ranges[mpi_rank];
  // _local_range.second = _all_ranges[mpi_rank + 1];
}
//-----------------------------------------------------------------------------
void RangeMap::set_local_to_global(std::vector<std::size_t> indices)
{
  dolfin_assert(_local_size > 0);
  _local_to_global = indices;

  const std::size_t mpi_rank = MPI::rank(_mpi_comm);
  for (const auto &node : _local_to_global)
  {
    const std::size_t p = std::upper_bound(_all_ranges.begin(),
                                           _all_ranges.end(), node)
                                - _all_ranges.begin() - 1;
    dolfin_assert(p != mpi_rank);
    _off_process_owner.push_back(p);
  }
}
//-----------------------------------------------------------------------------
