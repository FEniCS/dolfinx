// Copyright (C) 2015-2018 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
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
  /// Create Index map with local_size owned blocks on this process, and blocks
  /// have size block_size.
  ///
  /// Collective
  IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
           const std::vector<std::size_t>& ghosts, std::size_t block_size);

  /// Create Index map with local_size owned blocks on this process, and blocks
  /// have size block_size.
  ///
  /// Collective
  IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
           const std::vector<std::int64_t>& ghosts, std::size_t block_size);

  /// Copy constructor
  IndexMap(const IndexMap& map) = default;

  /// Move constructor
  IndexMap(IndexMap&& map) = default;

  /// Destructor
  ~IndexMap() = default;

  /// Range of indices (global) owned by this process
  std::array<std::int64_t, 2> local_range() const;

  /// Block size
  int block_size() const;

  /// Number of ghost indices on this process
  std::int32_t num_ghosts() const;

  /// Number of indices owned by on this process
  std::int32_t size_local() const;

  /// Number indices across communicator
  std::int64_t size_global() const;

  /// Local-to-global map for ghosts (local indexing beyond end of local
  /// range)
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghosts() const;

  /// Get global index for local index i (index of the block)
  std::int64_t local_to_global(std::int64_t local_index) const
  {
    assert(local_index >= 0);
    const std::int64_t local_size
        = _all_ranges[_myrank + 1] - _all_ranges[_myrank];
    if (local_index < local_size)
    {
      const std::int64_t global_offset = _all_ranges[_myrank];
      return global_offset + local_index;
    }
    else
    {
      assert((local_index - local_size) < _ghosts.size());
      return _ghosts[local_index - local_size];
    }
  }

  /// Owner rank of each ghost entry
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& ghost_owners() const;

  /// Get process that owns index (global block index)
  int owner(std::int64_t global_index) const;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as size_local().
  void scatter_fwd(const std::vector<std::int64_t>& local_data,
                   std::vector<std::int64_t>& remote_data, int n) const;
  void scatter_fwd(const std::vector<std::int32_t>& local_data,
                   std::vector<std::int32_t>& remote_data, int n) const;

  std::vector<std::int64_t>
  scatter_fwd(const std::vector<std::int64_t>& local_data, int n) const;
  std::vector<std::int32_t>
  scatter_fwd(const std::vector<std::int32_t>& local_data, int n) const;

  /// Send n values for each ghost index to owning to processes. The size
  /// of the input array remote_data must be the same as num_ghosts().
  void scatter_rev(std::vector<std::int64_t>& local_data,
                   const std::vector<std::int64_t>& remote_data, int n,
                   MPI_Op op) const;
  void scatter_rev(std::vector<std::int32_t>& local_data,
                   const std::vector<std::int32_t>& remote_data, int n,
                   MPI_Op op) const;

private:
  // MPI Communicator
  // dolfin::MPI::Comm _mpi_comm;
  MPI_Comm _mpi_comm;

  // Cache rank on mpi_comm (otherwise calls to MPI_Comm_rank can be
  // excessive)
  int _myrank;

public:
  // FIXME: This could get big for large process counts
  // Range of ownership of index for all processes
  std::vector<std::int64_t> _all_ranges;

private:
  // Local-to-global map for ghost indices
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _ghosts;

  // Owning process for each ghost index
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _ghost_owners;

  // Block size
  int _block_size;

  template <typename T>
  void scatter_fwd_impl(const std::vector<T>& local_data,
                        std::vector<T>& remote_data, int n) const;
  template <typename T>
  void scatter_rev_impl(std::vector<T>& local_data,
                        const std::vector<T>& remote_data, int n,
                        MPI_Op op) const;
};

} // namespace common
} // namespace dolfin
