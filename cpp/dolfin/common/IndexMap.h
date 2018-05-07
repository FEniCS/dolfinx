// Copyright (C) 2015-2018 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
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
/// from the local block indices  to a unique global block index.

class IndexMap
{
public:
  /// MapSize (ALL=(all indices in this map), OWNED=(indices owned by this
  /// process), GHOST=(ghost (unowned) local indices), GLOBAL=(total indices
  /// on communicator)
  enum class MapSize : int32_t
  {
    ALL = 0,
    OWNED = 1,
    GHOSTS = 2,
    GLOBAL = 3
  };

  /// Create Index map with local_size owned blocks on this process, and blocks
  /// have size block_size.
  ///
  /// Collective
  IndexMap(MPI_Comm mpi_comm, std::size_t local_size,
           const std::vector<std::size_t>& ghosts, std::size_t block_size);

  /// Copy constructor
  IndexMap(const IndexMap& map) = default;

  /// Move constructor
  IndexMap(IndexMap&& map) = default;

  /// Destructor
  ~IndexMap() = default;

  /// Range of indices owned by this process
  std::array<std::int64_t, 2> local_range() const;

  /// Block size
  int block_size() const;

  /// Size of set (MapSize::OWNED, MapSize::GHOSTS, MapSize::ALL or
  /// MapSize::GLOBAL)
  std::size_t size(MapSize type) const;

  /// Local-to-global map for ghosts (local indexing beyond end of local
  /// range)
  const EigenArrayXi64& ghosts() const;

  /// Get global index for local index i
  std::size_t local_to_global(std::size_t i) const;

  // TODO: remove
  /// Local to global index
  std::size_t local_to_global_index(std::size_t i) const;

  /// Owners of ghost entries
  const EigenArrayXi32& ghost_owners() const;

  /// Get process that owns index (global index)
  int owner(std::size_t global_index) const;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

private:
  // MPI Communicator
  dolfin::MPI::Comm _mpi_comm;

  // Cache rank on mpi_comm (otherwise calls to MPI_Comm_rank can be
  // excessive)
  int _myrank;

public:
  // FIXME: This could get big for large process counts
  // Range of ownership of index for all processes
  std::vector<std::size_t> _all_ranges;

private:
  // Local-to-global map for ghost indices
  EigenArrayXi64 _ghosts;

  // Owning process for each ghost index
  EigenArrayXi32 _ghost_owners;

  // Block size
  int _block_size;
};

// Function which may appear in a hot loop
inline std::size_t IndexMap::local_to_global(std::size_t i) const
{
  const std::size_t local_size
      = _all_ranges[_myrank + 1] - _all_ranges[_myrank];
  const std::size_t global_offset = _all_ranges[_myrank];

  if (i < local_size)
    return (i + global_offset);
  else
    return _ghosts[i - local_size];
}

// Function which may appear in a hot loop
inline std::size_t IndexMap::local_to_global_index(std::size_t i) const
{
  const std::size_t local_size
      = _block_size * (_all_ranges[_myrank + 1] - _all_ranges[_myrank]);
  const std::size_t global_offset = _block_size * _all_ranges[_myrank];

  if (i < local_size)
    return (i + global_offset);
  else
  {
    const std::div_t div = std::div((i - local_size), _block_size);
    const int component = div.rem;
    const int index = div.quot;
    assert(index < _ghosts.size());
    return _block_size * _ghosts[index] + component;
  }
}
} // namespace common
} // namespace dolfin
