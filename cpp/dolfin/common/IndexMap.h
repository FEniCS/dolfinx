// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/MPI.h>
#include <vector>

namespace dolfin
{

namespace common
{

/// This class represents the distribution index arrays across
/// processes. An index array is a contiguous collection of N+1
/// block indices [0, 1, . . ., N] that are distributed across processes M
/// processes. On a given process, the IndexMap stores a portion of
/// the index set using local indices [0, 1, . . . , n], and a map
/// from the local block indices to a unique global block index.

class IndexMap
{
public:
  /// MapSize (ALL=(all local indices), OWNED=(owned local indices),
  /// UNOWNED=(unowned local indices), GLOBAL=(total indices
  /// globally)
  enum class MapSize : int32_t
  {
    ALL = 0,
    OWNED = 1,
    UNOWNED = 2,
    GLOBAL = 3
  };

  /// Create Index map with local_size owned blocks on this process, and blocks
  /// have size block_size.
  ///
  /// Collective
  IndexMap(MPI_Comm mpi_comm, std::size_t local_size, std::size_t block_size);

  /// Copy constructor
  IndexMap(const IndexMap& map) = default;

  /// Move constructor
  IndexMap(IndexMap&& map) = default;

  /// Destructor
  ~IndexMap() = default;

  /// Local range of block indices
  std::array<std::int64_t, 2> local_range() const;

  /// Get number of local blocks of type MapSize::OWNED,
  /// MapSize::UNOWNED, MapSize::ALL or MapSize::GLOBAL
  std::size_t size(MapSize type) const;

  /// Get local to global map for unowned blocks
  /// (local indexing beyond end of local range)
  const std::vector<std::size_t>& local_to_global_unowned() const;

  /// Get global block index of local block i
  std::size_t local_to_global(std::size_t i) const;

  /// Local to global index
  std::size_t local_to_global_index(std::size_t i) const;

  /// Set local_to_global map for unowned blocks (beyond end of local
  /// range). Computes and stores off-process owner array.
  void set_block_local_to_global(const std::vector<std::size_t>& indices);

  /// Get off process owner for unowned blocks
  const std::vector<int>& block_off_process_owner() const;

  /// Get process owner of any global index
  int global_block_index_owner(std::size_t index) const;

  /// Get block size
  int block_size() const;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

private:
  // MPI Communicator
  dolfin::MPI::Comm _mpi_comm;

  // Cache rank of mpi_comm (otherwise calls to MPI_Comm_rank can be
  // excessive)
  unsigned int _rank;

  // FIXME: This could get big for large process counts
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
  // These two calls get hopefully optimized out of hot loops due to
  // inlining
  const std::size_t local_size = size(IndexMap::MapSize::OWNED);
  const std::size_t global_offset = local_range()[0];

  if (i < local_size)
    return (i + global_offset);
  else
    return _local_to_global[i - local_size];
}

// Function which may appear in a hot loop
inline std::size_t IndexMap::local_to_global_index(std::size_t i) const
{
  // These two calls get hopefully optimized out of hot loops due to
  // inlining
  const std::size_t local_size = _block_size * size(IndexMap::MapSize::OWNED);
  const std::size_t global_offset = _block_size * local_range()[0];

  if (i < local_size)
    return (i + global_offset);
  else
  {
    const std::div_t div = std::div((i - local_size), _block_size);
    const int component = div.rem;
    const int index = div.quot;
    assert((std::size_t)index < _local_to_global.size());
    return _block_size * _local_to_global[index] + component;
  }
}
}
}
