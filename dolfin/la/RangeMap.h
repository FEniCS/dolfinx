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
// First added:  2010-11-16
// Last changed: 2010-11-25

#ifndef __RANGE_MAP_H
#define __RANGE_MAP_H

#include <limits>
#include <vector>
#include <dolfin/common/MPI.h>

namespace dolfin
{

  class RangeMap
  {
  public:
    /// Constructor
    RangeMap()
    {}

    /// Range map with no data
    explicit RangeMap(MPI_Comm mpi_comm)
      : _mpi_comm(mpi_comm)
    {}

    /// Range map with local size on each process
    RangeMap(MPI_Comm mpi_comm, std::size_t local_size, std::size_t block_size)
      : _mpi_comm(mpi_comm)
    {
      init(local_size, block_size);
    }

    /// Destructor
    ~RangeMap()
    {}

    /// Initialise with local_size
    void init(std::size_t local_size, std::size_t block_size);

    /// Local range of indices
    std::pair<std::size_t, std::size_t> local_range() const
    {
      const std::size_t rank = MPI::rank(_mpi_comm);

      // Uninitialised
      if(_all_ranges.size() == 0)
      {
        warning("Asking for size of uninitialised range\n");
        return std::pair<std::size_t, std::size_t>(0, 0);
      }

      return std::make_pair(_block_size*_all_ranges[rank],
                            _block_size*_all_ranges[rank + 1]);
    }

    /// Number of local indices
    std::size_t size() const
    {
      const std::size_t rank = MPI::rank(_mpi_comm);

      // Uninitialised
      if(_all_ranges.size() == 0)
      {
        warning("Asking for size of uninitialised range\n");
        return 0;
      }


      return _block_size*(_all_ranges[rank + 1]
                          - _all_ranges[rank]);
    }

    /// Global size of map
    std::size_t size_global() const
    { return _all_ranges.back();  }

    /// Get local to global map for unowned indices
    const std::vector<std::size_t>& local_to_global_unowned() const
    { return _local_to_global; }

    /// Get global index of local index i
    std::size_t local_to_global(std::size_t i) const
    {
      const std::size_t local_size = size();
      const std::size_t global_offset = local_range().first;

      if (i < size())
        return (i + global_offset);
      else
      {
        const std::div_t div = std::div((i - local_size),
                                        _block_size);
        const int component = div.rem;
        const int index = div.quot;
        dolfin_assert((std::size_t) index < _local_to_global.size());
        return _block_size*_local_to_global[index] + component;
      }
    }

    /// Set local_to_global map for unowned indices
    void set_local_to_global(std::vector<std::size_t>& indices);

    /// Get off process owner
    const std::vector<int>& off_process_owner() const
    {
      return _off_process_owner;
    }

    /// Return MPI communicator
    MPI_Comm mpi_comm() const
    { return _mpi_comm; }

  private:

    // MPI Communicator
    MPI_Comm _mpi_comm;

    // Range of ownership of index for all processes
    std::vector<std::size_t> _all_ranges;

    // Local to global map for off-process entries
    std::vector<std::size_t> _local_to_global;

    // Off process owner cache
    std::vector<int> _off_process_owner;

    // Block size
    int _block_size;

  };
}

#endif
