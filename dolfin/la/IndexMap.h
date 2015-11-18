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

    /// Constructor
    IndexMap();

    /// Range map with no data
    explicit IndexMap(MPI_Comm mpi_comm);

    /// Range map with local size on each process
    IndexMap(MPI_Comm mpi_comm, std::size_t local_size, std::size_t block_size);

    /// Destructor
    ~IndexMap();

    /// Initialise with local_size
    void init(std::size_t local_size, std::size_t block_size);

    /// Local range of indices
    std::pair<std::size_t, std::size_t> local_range() const;

    /// Number of local indices
    std::size_t size() const;

    /// Number of local indices "local", "shared" or "all"
    std::size_t size(const std::string type) const;

    /// Global size of map
    std::size_t size_global() const;

    /// Get local to global map for unowned indices
    const std::vector<std::size_t>& local_to_global_unowned() const;

    /// Get global index of local index i
    std::size_t local_to_global(std::size_t i) const;

    /// Set local_to_global map for unowned indices
    void set_local_to_global(std::vector<std::size_t>& indices);

    /// Get off process owner
    const std::vector<int>& off_process_owner() const;

    /// Get block size
    int block_size() const;

    /// Return MPI communicator
    MPI_Comm mpi_comm() const;

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
