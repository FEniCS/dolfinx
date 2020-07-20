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
#include <tuple>
#include <vector>

namespace dolfinx::common
{
// Forward declaration
class IndexMap;

/// Compute layout data and ghost indices for a stacked (concatenated)
/// index map, i.e. 'splice' multiple maps into one. Communication is
/// required to compute the new ghost indices.
///
/// @param[in] maps List of index maps
/// @returns The (0) global offset of a stacked map for this rank, (1)
///   local offset for each submap in the stacked map, and (2) new
///   indices for the ghosts for each submap (3) owner rank of each ghost
///   entry for each submap
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
stack_index_maps(
    const std::vector<std::reference_wrapper<const common::IndexMap>>& maps);

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

  /// Edge directions of neighborhood communicator
  enum class Direction
  {
    reverse,  // Ghost to owner
    forward,  // Owner to ghost
    symmetric // Symmetric. NOTE: To be removed
  };

  /// Create an non-overlapping index map with local_size owned blocks on this
  /// process.
  ///
  /// @note Collective
  /// @param[in] comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMap, i.e. the number
  ///   of owned entries
  /// @param[in] block_size The block size of the IndexMap
  IndexMap(MPI_Comm comm, std::int32_t local_size, int block_size);

  /// Create an index map with local_size owned blocks on this process, and
  /// blocks have size block_size.
  ///
  /// @note Collective
  /// @param[in] mpi_comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMap, i.e. the number
  ///   of owned entries
  /// @param[in] dest_ranks Ranks that 'ghost' indices that are owned by
  ///   the calling rank. I.e., ranks that the caller will send data to
  ///   when updating ghost values.
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] src_ranks Owner rank (on global communicator) of each
  ///   entry in @p ghosts
  /// @param[in] block_size The block size of the IndexMap
  IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
           const std::vector<int>& dest_ranks,
           const std::vector<std::int64_t>& ghosts,
           const std::vector<int>& src_ranks, int block_size);

  /// Create an index map
  ///
  /// @note Collective
  /// @param[in] mpi_comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMap, i.e. the number
  ///   of owned entries
  /// @param[in] dest_ranks Ranks that ghost indices owned by the
  ///   calling rank
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] src_ranks Owner rank (on global communicator) of each
  ///   ghost entry
  /// @param[in] block_size The block size of the IndexMap
  IndexMap(
      MPI_Comm mpi_comm, std::int32_t local_size,
      const std::vector<int>& dest_ranks,
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
          ghosts,
      const std::vector<int>& src_ranks, int block_size);

  /// Copy constructor
  IndexMap(const IndexMap& map) = delete;

  /// Move constructor
  IndexMap(IndexMap&& map) = default;

  /// Destructor
  ~IndexMap() = default;

  /// Range of indices (global) owned by this process
  std::array<std::int64_t, 2> local_range() const noexcept;

  /// Block size
  int block_size() const noexcept;

  /// Number of ghost indices on this process
  std::int32_t num_ghosts() const;

  /// Number of indices owned by on this process
  std::int32_t size_local() const;

  /// Number indices across communicator
  std::int64_t size_global() const;

  /// Local-to-global map for ghosts (local indexing beyond end of local
  /// range)
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts() const;

  /// Return a MPI communicator with attached distributed graph topology
  /// information
  /// @param[in] dir Edge direction of communicator (forward, reverse,
  /// symmetric)
  /// @return A neighborhood communicator for the specified edge direction
  MPI_Comm comm(Direction dir = Direction::symmetric) const;

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

  /// @todo Consider removing this function in favour of the version
  /// that accepts an Eigen array.
  ///
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
  ///   Returns -1 if the local index does not exist on this process.
  std::vector<std::int32_t>
  global_to_local(const std::vector<std::int64_t>& indices,
                  bool blocked = true) const;

  /// Compute local indices for array of global indices
  /// @param[in] indices Global indices
  /// @param[in] blocked If true work with blocked indices. If false the
  ///   input indices are not block-wise.
  /// @return The local of the corresponding global index in indices.
  ///   Return -1 if the local index does not exist on this process.
  std::vector<std::int32_t> global_to_local(
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
          indices,
      bool blocked = true) const;

  /// Global indices
  /// @return The global index for all local indices (0, 1, 2, ...) on
  ///   this process, including ghosts
  std::vector<std::int64_t> global_indices(bool blocked = true) const;

  /// @todo Reconsider name
  /// Local (owned) indices shared with neighbor processes, i.e. are
  /// ghosts on other processes
  /// @return List of indices that are ghosted on other processes
  const std::vector<std::int32_t>& shared_indices() const;

  /// Owner rank (on global communicator) of each ghost entry
  Eigen::Array<int, Eigen::Dynamic, 1> ghost_owner_rank() const;

  /// Return array of global indices for all indices on this process,
  /// including ghosts
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1>
  indices(bool unroll_block) const;

  /// @todo Aim to remove this function? If it's kept, should it work
  /// with neighborhood ranks?
  ///
  /// Compute map from each local (owned) index to the set of ranks that
  /// have the index as a ghost
  /// @return shared indices
  std::map<std::int32_t, std::set<int>> compute_shared_indices() const;

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as n * size_local().
  ///
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
  ///
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
  ///
  /// @param[in] local_data Local data associated with each owned local
  ///   index to be sent to process where the data is ghosted. Size must
  ///   be n * size_local().
  /// @param[in] n Number of data items per index
  /// @return Ghost data on this process received from the owning
  ///   process. Size will be n * num_ghosts().
  std::vector<std::int64_t>
  scatter_fwd(const std::vector<std::int64_t>& local_data, int n) const;

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost
  ///
  /// @param[in] local_data Local data associated with each owned local
  ///   index to be sent to process where the data is ghosted. Size must
  ///   be n * size_local().
  /// @param[in] n Number of data items per index
  /// @return Ghost data on this process received from the owning
  ///   process. Size will be n * num_ghosts().
  std::vector<std::int32_t>
  scatter_fwd(const std::vector<std::int32_t>& local_data, int n) const;

  /// Send n values for each ghost index to owning to the process
  ///
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

  /// Send n values for each ghost index to owning to the process
  ///
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
  int _block_size;

  // Range of indices (global) owned by this process
  std::array<std::int64_t, 2> _local_range;

  // Number indices across communicator
  std::int64_t _size_global;

  // MPI neighborhood communicators

  // Communicator where the source ranks own the indices in the callers
  // halo, and the destination ranks 'ghost' indices owned by the
  // caller. I.e.,
  // - in-edges (src) are from ranks that own my ghosts
  // - out-edges (dest) go to ranks that 'ghost' my owned indices
  dolfinx::MPI::Comm _comm_owner_to_ghost;

  // Communicator where the source ranks have ghost indices that are
  // owned by the caller, and the destination ranks are the owners of
  // indices in the callers halo region. I.e.,
  // - in-edges (src) are from ranks that 'ghost' my owned indicies
  // - out-edges (dest) are to the owning ranks of my ghost indices
  dolfinx::MPI::Comm _comm_ghost_to_owner;

  // TODO: remove
  dolfinx::MPI::Comm _comm_symmetric;

  // Local-to-global map for ghost indices
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _ghosts;

  // Owning neighborhood rank (out edge) on '_comm_owner_to_ghost'
  // communicator for each ghost index
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _ghost_owners;

  // TODO: replace _shared_disp and _shared_disp by an AdjacencyList

  // TODO: _shared_indices are received on _comm_ghost_to_owner, and
  // _shared_indices is the recv_disp on _comm_ghost_to_owner. Check for
  // corect use on _comm_owner_to_ghost. Can guarantee that
  // _comm_owner_to_ghost and _comm_ghost_to_owner are the transpose of
  // each other?

  // Owned local indices that are in the halo (ghost) region on other
  // ranks
  std::vector<std::int32_t> _shared_indices;

  // FIXME: explain better the ranks
  // Displacement vector for _shared_indices. _shared_indices[i] is the
  // starting postion in _shared_indices for data that is ghosted on
  // rank i, where i is the ith outgoing edge on _comm_owner_to_ghost.
  std::vector<std::int32_t> _shared_disp;

  template <typename T>
  void scatter_fwd_impl(const std::vector<T>& local_data,
                        std::vector<T>& remote_data, int n) const;
  template <typename T>
  void scatter_rev_impl(std::vector<T>& local_data,
                        const std::vector<T>& remote_data, int n,
                        Mode op) const;
};

} // namespace dolfinx::common
