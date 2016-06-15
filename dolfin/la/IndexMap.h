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

#ifndef __INDEX_MAP_H
#define __INDEX_MAP_H

#include <utility>
#include <vector>
#include <dolfin/common/MPI.h>

namespace dolfin
{

  /// This class represents the distribution index arrays across
  /// processes. An index array is a contiguous collection of N+1
  /// indices [0, 1, . . ., N] that are distributed across processes M
  /// processes. On a given process, the IndexMap stores a portion of
  /// the index set using local indices [0, 1, . . . , n], and a map
  /// from the local indices to a unique global index.

  class IndexMap
  {
  public:

    /// Size required: number of ALL local indices, only OWNED indices, only UNOWNED,
    /// or the GLOBAL number of indices
    enum class MapSize : int32_t { ALL = 0,
                                   OWNED = 1,
                                   UNOWNED = 2,
                                   GLOBAL = 3 };

    /// Constructor
    IndexMap();

    /// Index map with no data
    explicit IndexMap(MPI_Comm mpi_comm);

    /// Index map with local size on each process. This constructor
    /// is collective
    IndexMap(MPI_Comm mpi_comm, std::size_t local_size, std::size_t block_size);

    /// Destructor
    ~IndexMap();

    /// Initialise with number of local entries and block size. This
    /// function is collective
    void init(std::size_t local_size, std::size_t block_size);

    /// Local range of indices
    std::pair<std::size_t, std::size_t> local_range() const;

    /// Get number of local indices of type MapSize::OWNED,
    /// MapSize::UNOWNED, MapSize::ALL or MapSize::GLOBAL
    std::size_t size(MapSize type) const;

    /// Get local to global map for unowned indices
    /// (local indexing beyond end of local range)
    const std::vector<std::size_t>& local_to_global_unowned() const;

    /// Get global index of local index i
    std::size_t local_to_global(std::size_t i) const;

    /// Set local_to_global map for unowned indices (beyond end of local
    /// range). Computes and stores off-process owner array.
    void set_local_to_global(const std::vector<std::size_t>& indices);

    /// Get off process owner for unowned indices
    const std::vector<int>& off_process_owner() const;

    /// Get block size
    int block_size() const;

    /// Return MPI communicator
    MPI_Comm mpi_comm() const;

  private:

    // MPI Communicator
    MPI_Comm _mpi_comm;

    // Cache rank of mpi_comm (otherwise calls to MPI_Comm_rank can be excessive)
    unsigned int _rank;

    // Range of ownership of index for all processes
    std::vector<std::size_t> _all_ranges;

    // Local to global map for off-process entries
    std::vector<std::size_t> _local_to_global;

    // Off process owner cache
    std::vector<int> _off_process_owner;

    // Block size
    int _block_size;

  };


  // Function which may appear in a hot loop
  inline std::size_t IndexMap::local_to_global(std::size_t i) const
  {
    // These two calls get hepefully optimized out of hot loops due
    // to inlining
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

}

#endif
