// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <map>
#include <set>
#include <vector>

namespace dolfinx
{

namespace common
{

/// This class represents the distribution index arrays across
/// processes. An index array is a contiguous collection of N+1 block
/// indices [0, 1, . . ., N] that are distributed across processes M
/// processes. On a given process, the IndexMap stores a portion of the
/// index set using local indices [0, 1, . . . , n], and a map from the
/// local block indices  to a unique global block index.

class IndexMap
{
public:
  /// Mode for reverse scatter operation
  enum class Mode
  {
    insert,
    add
  };

  /// Create Index map with local_size owned blocks on this process, and
  /// blocks have size block_size.
  ///
  /// Collective
  /// @param[in] mpi_comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMap, i.e. the number
  ///   of owned entries
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] block_size The block size of the IndexMap
  IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
           const std::vector<std::int64_t>& ghosts, int block_size);

  /// Create Index map with local_size owned blocks on this process, and
  /// blocks have size block_size.
  ///
  /// Collective
  /// @param[in] mpi_comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMap, i.e. the number
  ///   of owned entries
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] block_size The block size of the IndexMap
  IndexMap(
      MPI_Comm mpi_comm, std::int32_t local_size,
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
          ghosts,
      int block_size);

  /// Copy constructor
  IndexMap(const IndexMap& map) = delete;

  /// Move constructor
  IndexMap(IndexMap&& map) = default;

  /// Destructor
  ~IndexMap();

  /// Range of indices (global) owned by this process
  std::array<std::int64_t, 2> local_range() const;

  /// Block size
  const int block_size;

  /// Number of ghost indices on this process
  std::int32_t num_ghosts() const;

  /// Number of indices owned by on this process
  std::int32_t size_local() const;

  /// Number indices across communicator
  std::int64_t size_global() const;

  /// Local-to-global map for ghosts (local indexing beyond end of local
  /// range)
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts() const;

  /// Compute global indices for array of local indices
  /// @param[in] indices Local indices
  /// @param[in] blocked If true work with blocked indices. If false the
  ///   input indices are not block-wise.
  /// @return The global index of the corresponding local index in
  ///   indices.
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global(
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&
          indices,
      bool blocked = true) const;

  /// Compute global indices for array of local indices
  /// @param[in] indices Local indices
  /// @param[in] blocked If true work with blocked indices. If false the
  ///   input indices are not block-wise.
  /// @return The global index of the corresponding local index in
  ///   indices.
  std::vector<std::int64_t>
  local_to_global(const std::vector<std::int32_t>& indices,
                  bool blocked = true) const;

  /// Compute local indices for array of global indices
  /// @param[in] indices Global indices
  /// @param[in] blocked If true work with blocked indices. If false the
  ///   input indices are not block-wise.
  /// @return The local of the corresponding global index in indices.
  ///   Return -1 if the local index does not exist on this process.
  std::vector<std::int32_t>
  global_to_local(const std::vector<std::int64_t>& indices,
                  bool blocked = true) const;

  /// Global indices
  /// @return The global index for all local indices (0, 1, 2, ...) on this
  /// process, including ghosts
  std::vector<std::int64_t> global_indices(bool blocked = true) const;

  /// @todo Remove this function
  /// Get global index for local index i (index of the block)
  std::int64_t local_to_global(std::int32_t local_index) const
  {
    assert(local_index >= 0);
    const std::int32_t local_size
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

  /// @todo Reconsider name
  /// Local (owned) indices shared with neighbour processes, i.e. are
  /// ghosts on other processes
  /// @return List of indices that are ghosted on other processes
  const std::vector<std::int32_t>& forward_indices() const
  {
    return _forward_indices;
  }

  /// Owner rank (on global communicator) of each ghost entry
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_owners() const;

  /// Get process that owns index (global block index)
  int owner(std::int64_t global_index) const;

  /// Return array of global indices for all indices on this process,
  /// including ghosts
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1>
  indices(bool unroll_block) const;

  /// Return MPI communicator
  /// @return The communicator on which the IndexMap is defined
  MPI_Comm mpi_comm() const;

  /// Return MPI neighbourhood communicator
  /// @return The neighbourhood communicator
  MPI_Comm mpi_comm_neighborhood() const;

  /// Compute map from each local index to the complete set of sharing processes
  /// for that index
  /// @return shared indices
  std::map<std::int32_t, std::set<std::int32_t>> compute_shared_indices() const;

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as n * size_local().
  /// @param[in] local_data Local data associated with each owned local
  ///   index to be sent to process where the data is ghosted. Size must
  ///   be n * size_local().
  /// @param[in,out] remote_data Ghost data on this process received
  ///   from the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  void scatter_fwd(const std::vector<std::int64_t>& local_data,
                   std::vector<std::int64_t>& remote_data, int n) const;

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as n * size_local().
  /// @param[in] local_data Local data associated with each owned local
  ///   index to be sent to process where the data is ghosted. Size must
  ///   be n * size_local().
  /// @param[in,out] remote_data Ghost data on this process received
  ///   from the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  void scatter_fwd(const std::vector<std::int32_t>& local_data,
                   std::vector<std::int32_t>& remote_data, int n) const;

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as n * size_local().
  /// @param[in] local_data Local data associated with each owned local
  ///   index to be sent to process where the data is ghosted. Size must
  ///   be n * size_local().
  /// @param[in] n Number of data items per index
  /// @return Ghost data on this process received from the owning
  ///   process. Size will be n * num_ghosts().
  std::vector<std::int64_t>
  scatter_fwd(const std::vector<std::int64_t>& local_data, int n) const;

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost.
  /// @param[in] local_data Local data associated with each owned local
  ///   index to be sent to process where the data is ghosted. Size must
  ///   be n * size_local().
  /// @param[in] n Number of data items per index
  /// @return Ghost data on this process received from the owning
  ///   process. Size will be n * num_ghosts().
  std::vector<std::int32_t>
  scatter_fwd(const std::vector<std::int32_t>& local_data, int n) const;

  /// Send n values for each ghost index to owning to the process.
  /// @param[in,out] local_data Local data associated with each owned
  ///   local index to be sent to process where the data is ghosted.
  ///   Size must be n * size_local().
  /// @param[in] remote_data Ghost data on this process received from
  ///   the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  /// @param[in] op Sum or set received values in local_data
  void scatter_rev(std::vector<std::int64_t>& local_data,
                   const std::vector<std::int64_t>& remote_data, int n,
                   IndexMap::Mode op) const;

  /// Send n values for each ghost index to owning to the process.
  /// @param[in,out] local_data Local data associated with each owned
  ///   local index to be sent to process where the data is ghosted.
  ///   Size must be n * size_local().
  /// @param[in] remote_data Ghost data on this process received from
  ///   the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  /// @param[in] op Sum or set received values in local_data
  void scatter_rev(std::vector<std::int32_t>& local_data,
                   const std::vector<std::int32_t>& remote_data, int n,
                   IndexMap::Mode op) const;

private:
  // MPI Communicator
  MPI_Comm _mpi_comm;

  // MPI Communicator for neighbourhood only
  MPI_Comm _neighbour_comm;

  // Cache rank on mpi_comm (otherwise calls to MPI_Comm_rank can be
  // excessive)
  int _myrank;

public:
  // FIXME: This could get big for large process counts
  // Range of ownership of index for all processes
  std::vector<std::int64_t> _all_ranges;

private:
  // Local-to-global map for ghost indices
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _ghosts;

  // Owning neighbour for each ghost index
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _ghost_owners;

  // Number of indices to send to each neighbour process (ghost ->
  // owner, i.e. forward mode scatter)
  std::vector<std::int32_t> _forward_sizes;

  // "Owned" local indices shared with neighbour processes
  std::vector<std::int32_t> _forward_indices;

  template <typename T>
  void scatter_fwd_impl(const std::vector<T>& local_data,
                        std::vector<T>& remote_data, int n) const;
  template <typename T>
  void scatter_rev_impl(std::vector<T>& local_data,
                        const std::vector<T>& remote_data, int n,
                        Mode op) const;
};

} // namespace common
} // namespace dolfinx
