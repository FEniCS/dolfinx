// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::common
{
// Forward declaration
class IndexMap;

/// Given a vector of indices (local numbering, owned or ghost) and an
/// index map, this function returns the indices owned by this process,
/// including indices that might have been in the list of indices on
/// another processes.
/// @param[in] indices List of indices
/// @param[in] map The index map
/// @return Vector of indices owned by the process
std::vector<int32_t>
compute_owned_indices(const xtl::span<const std::int32_t>& indices,
                      const IndexMap& map);

/// Compute layout data and ghost indices for a stacked (concatenated)
/// index map, i.e. 'splice' multiple maps into one. Communication is
/// required to compute the new ghost indices.
///
/// @param[in] maps List of (index map, block size) pairs
/// @returns The (0) global offset of a stacked map for this rank, (1)
/// local offset for each submap in the stacked map, and (2) new indices
/// for the ghosts for each submap (3) owner rank of each ghost entry
/// for each submap
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
stack_index_maps(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// This class represents the distribution index arrays across
/// processes. An index array is a contiguous collection of N+1 indices
/// [0, 1, . . ., N] that are distributed across M processes. On a given
/// process, the IndexMap stores a portion of the index set using local
/// indices [0, 1, . . . , n], and a map from the local indices to a
/// unique global index.

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
    reverse, // Ghost to owner
    forward, // Owner to ghost
  };

  /// Create an non-overlapping index map with local_size owned on this
  /// process.
  ///
  /// @note Collective
  /// @param[in] comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMap, i.e. the number
  /// of owned entries
  IndexMap(MPI_Comm comm, std::int32_t local_size);

  /// Create an index map with local_size owned indiced on this process
  ///
  /// @note Collective
  /// @param[in] comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMap, i.e. the number
  /// of owned entries
  /// @param[in] dest_ranks Ranks that 'ghost' indices that are owned by
  /// the calling rank. I.e., ranks that the caller will send data to
  /// when updating ghost values.
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] src_ranks Owner rank (on global communicator) of each
  /// entry in @p ghosts
  IndexMap(MPI_Comm comm, std::int32_t local_size,
           const xtl::span<const int>& dest_ranks,
           const xtl::span<const std::int64_t>& ghosts,
           const xtl::span<const int>& src_ranks);

private:
  template <typename U, typename V, typename W, typename X>
  IndexMap(std::array<std::int64_t, 2> local_range, std::size_t size_global,
           MPI_Comm comm, U&& comm_owner_to_ghost, U&& comm_ghost_to_owner,
           V&& displs_recv_fwd, V&& ghost_pos_recv_fwd, W&& ghosts,
           X&& shared_indices)
      : _local_range(local_range), _size_global(size_global), _comm(comm),
        _comm_owner_to_ghost(std::forward<U>(comm_owner_to_ghost)),
        _comm_ghost_to_owner(std::forward<U>(comm_ghost_to_owner)),
        _displs_recv_fwd(std::forward<V>(displs_recv_fwd)),
        _ghost_pos_recv_fwd(std::forward<V>(ghost_pos_recv_fwd)),
        _ghosts(std::forward<W>(ghosts)),
        _shared_indices(std::forward<X>(shared_indices))
  {
    _sizes_recv_fwd.resize(_displs_recv_fwd.size() - 1, 0);
    std::adjacent_difference(_displs_recv_fwd.cbegin() + 1,
                             _displs_recv_fwd.cend(), _sizes_recv_fwd.begin());

    const std::vector<int32_t>& displs_send = _shared_indices->offsets();
    _sizes_send_fwd.resize(_shared_indices->num_nodes(), 0);
    std::adjacent_difference(displs_send.cbegin() + 1, displs_send.cend(),
                             _sizes_send_fwd.begin());
  }

public:
  // Copy constructor
  IndexMap(const IndexMap& map) = delete;

  /// Move constructor
  IndexMap(IndexMap&& map) = default;

  /// Destructor
  ~IndexMap() = default;

  /// Move assignment
  IndexMap& operator=(IndexMap&& map) = default;

  // Copy assignment
  IndexMap& operator=(const IndexMap& map) = delete;

  /// Range of indices (global) owned by this process
  std::array<std::int64_t, 2> local_range() const noexcept;

  /// Number of ghost indices on this process
  std::int32_t num_ghosts() const noexcept;

  /// Number of indices owned by on this process
  std::int32_t size_local() const noexcept;

  /// Number indices across communicator
  std::int64_t size_global() const noexcept;

  /// Local-to-global map for ghosts (local indexing beyond end of local
  /// range)
  const std::vector<std::int64_t>& ghosts() const noexcept;

  /// Return the MPI communicator used to create the index map
  /// @return Communicator
  MPI_Comm comm() const;

  /// Return a MPI communicator with attached distributed graph topology
  /// information
  /// @param[in] dir Edge direction of communicator (forward, reverse)
  /// @return A neighborhood communicator for the specified edge direction
  MPI_Comm comm(Direction dir) const;

  /// Compute global indices for array of local indices
  /// @param[in] local Local indices
  /// @param[out] global The global indices
  void local_to_global(const xtl::span<const std::int32_t>& local,
                       const xtl::span<std::int64_t>& global) const;

  /// Compute local indices for array of global indices
  /// @param[in] global Global indices
  /// @param[out] local The local of the corresponding global index in 'global'.
  /// Returns -1 if the local index does not exist on this process.
  void global_to_local(const xtl::span<const std::int64_t>& global,
                       const xtl::span<std::int32_t>& local) const;

  /// Global indices
  /// @return The global index for all local indices (0, 1, 2, ...) on
  /// this process, including ghosts
  std::vector<std::int64_t> global_indices() const;

  /// Local (owned) indices shared with neighbor processes, i.e. are
  /// ghosts on other processes, grouped by sharing (neighbor) process
  /// (destination ranks in forward communicator and source ranks in the
  /// reverse communicator). `scatter_fwd_indices().links(p)` gives the
  /// list of owned indices that needs to be sent to neighbourhood rank
  /// `p` during a forward scatter.
  ///
  /// Entries are ordered such that `scatter_fwd_indices.offsets()` is
  /// the send displacement array for a forward scatter and
  /// `scatter_fwd_indices.array()[i]` in the index of the owned index
  /// that should be placed at position `i` in the send buffer for a
  /// forward scatter.
  /// @return List of indices that are ghosted on other processes
  const graph::AdjacencyList<std::int32_t>&
  scatter_fwd_indices() const noexcept;

  /// Position of ghost entries in the receive buffer after a forward
  /// scatter, e.g. for a receive buffer `b` and a set operation, the
  /// ghost values should be updated  by `ghost_value[i] =
  /// b[scatter_fwd_ghost_positions[i]]`.
  /// @return Position of the ith ghost entry in the received buffer
  const std::vector<std::int32_t>& scatter_fwd_ghost_positions() const noexcept;

  /// Owner rank on the global communicator of each ghost entry
  std::vector<int> ghost_owner_rank() const;

  /// Compute the owner on the neighborhood communicator of ghost indices
  std::vector<int> ghost_owner_neighbor_rank() const;

  /// @todo Aim to remove this function? If it's kept, should it work
  /// with neighborhood ranks?
  ///
  /// Compute map from each local (owned) index to the set of ranks that
  /// have the index as a ghost
  /// @return shared indices
  std::map<std::int32_t, std::set<int>> compute_shared_indices() const;

  /// Create new index map from a subset of indices in this index map.
  /// The order of the indices is preserved, with new map effectively a
  /// 'compressed' map.
  /// @param[in] indices Local indices in the map that should appear in
  /// the new index map. All indices must be owned, i.e. indices must be
  /// less than `this->size_local()`.
  /// @pre `indices` must be sorted and contain no duplicates
  /// @return The (i) new index map and (ii) a map from the ghost
  /// position in the new map to the ghost position in the original
  /// (this) map
  std::pair<IndexMap, std::vector<std::int32_t>>
  create_submap(const xtl::span<const std::int32_t>& indices) const;

  /// Start a non-blocking send of owned data to ranks that ghost the
  /// data. The communication is completed by calling
  /// IndexMap::scatter_fwd_end. The send and receive buffer should not
  /// be changed until after IndexMap::scatter_fwd_end has been called.
  ///
  /// @param[in] send_buffer Local data associated with each owned local
  /// index to be sent to process where the data is ghosted. It must not
  /// be changed until after a call to IndexMap::scatter_fwd_end. The
  /// order of data in the buffer is given by
  /// IndexMap::scatter_fwd_indices.
  /// @param data_type The MPI data type. To send data with a block size
  /// use `MPI_Type_contiguous` with size `n`
  /// @param request The MPI request handle for tracking the status of
  /// the non-blocking communication
  /// @param recv_buffer A buffer used for the received data. The
  /// position of ghost entries in the buffer is given by
  /// IndexMap::scatter_fwd_ghost_positions. The buffer must not be
  /// accessed or changed until after a call to
  /// IndexMap::scatter_fwd_end.
  template <typename T>
  void scatter_fwd_begin(const xtl::span<const T>& send_buffer,
                         MPI_Datatype& data_type, MPI_Request& request,
                         const xtl::span<T>& recv_buffer) const
  {
    // Send displacement
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();

    // Return early if there are no incoming or outgoing edges
    if (_displs_recv_fwd.size() == 1 and displs_send_fwd.size() == 1)
      return;

    // Get block size
    int n;
    MPI_Type_size(data_type, &n);
    n /= sizeof(T);
    if (static_cast<int>(send_buffer.size()) != n * displs_send_fwd.back())
      throw std::runtime_error("Incompatible send buffer size.");
    if (static_cast<int>(recv_buffer.size()) != n * _displs_recv_fwd.back())
      throw std::runtime_error("Incompatible receive buffer size..");

    // Start send/receive
    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_send_fwd.data(),
                            displs_send_fwd.data(), data_type,
                            recv_buffer.data(), _sizes_recv_fwd.data(),
                            _displs_recv_fwd.data(), data_type,
                            _comm_owner_to_ghost.comm(), &request);
  }

  /// Complete a non-blocking send from the local owner of to process
  /// ranks that have the index as a ghost. This function complete the
  /// communication started by IndexMap::scatter_fwd_begin.
  ///
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  void scatter_fwd_end(MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();
    if (_displs_recv_fwd.size() == 1 and displs_send_fwd.size() == 1)
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as n * size_local().
  ///
  /// @param[in] local_data Local data associated with each owned local
  /// index to be sent to process where the data is ghosted. Size must
  /// be n * size_local().
  /// @param[in,out] remote_data Ghost data on this process received
  /// from the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  template <typename T>
  void scatter_fwd(const xtl::span<const T>& local_data,
                   xtl::span<T> remote_data, int n) const
  {
    MPI_Datatype data_type;
    if (n == 1)
      data_type = dolfinx::MPI::mpi_type<T>();
    else
    {
      MPI_Type_contiguous(n, dolfinx::MPI::mpi_type<T>(), &data_type);
      MPI_Type_commit(&data_type);
    }

    const std::vector<std::int32_t>& indices = _shared_indices->array();
    std::vector<T> send_buffer(n * indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
      std::copy_n(std::next(local_data.cbegin(), n * indices[i]), n,
                  std::next(send_buffer.begin(), n * i));
    }

    MPI_Request request;
    std::vector<T> buffer_recv(n * _displs_recv_fwd.back());
    scatter_fwd_begin(xtl::span<const T>(send_buffer), data_type, request,
                      xtl::span<T>(buffer_recv));
    scatter_fwd_end(request);

    // Copy into ghost area ("remote_data")
    assert(remote_data.size() == n * _ghost_pos_recv_fwd.size());
    for (std::size_t i = 0; i < _ghost_pos_recv_fwd.size(); ++i)
    {
      std::copy_n(std::next(buffer_recv.cbegin(), n * _ghost_pos_recv_fwd[i]),
                  n, std::next(remote_data.begin(), n * i));
    }

    if (n != 1)
      MPI_Type_free(&data_type);
  }

  /// Start a non-blocking send of ghost values to the owning rank. The
  /// non-blocking communication is completed by calling
  /// IndexMap::scatter_rev_end. A reverse scatter is the transpose of
  /// IndexMap::scatter_fwd_begin.
  ///
  /// @param[in] send_buffer Send buffer filled with ghost data on this
  /// process to be sent to the owning rank. The order of the data is
  /// given by IndexMap::scatter_fwd_ghost_positions, with
  /// IndexMap::scatter_fwd_ghost_positions()[i] being the index of the
  /// ghost data that should be placed in position `i` of the buffer.
  /// @param data_type The MPI data type. To send data with a block size
  /// use `MPI_Type_contiguous` with size `n`
  /// @param request The MPI request handle for tracking the status of
  /// the send
  /// @param recv_buffer A buffer used for the received data. It must
  /// not be changed until after a call to IndexMap::scatter_rev_end.
  /// The ordering of the data is given by
  /// IndexMap::scatter_fwd_indices, with
  /// IndexMap::scatter_fwd_indices()[i] being the position in the owned
  /// data array that corresponds to position `i` in the buffer.
  template <typename T>
  void scatter_rev_begin(const xtl::span<const T>& send_buffer,
                         MPI_Datatype& data_type, MPI_Request& request,
                         const xtl::span<T>& recv_buffer) const
  {
    // Get displacement vector
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();

    // Return early if there are no incoming or outgoing edges
    if (_displs_recv_fwd.size() == 1 and displs_send_fwd.size() == 1)
      return;

    // Get block size
    int n;
    MPI_Type_size(data_type, &n);
    n /= sizeof(T);
    if (static_cast<int>(send_buffer.size()) != n * _ghosts.size())
      throw std::runtime_error("Inconsistent send buffer size.");
    if (static_cast<int>(recv_buffer.size()) != n * displs_send_fwd.back())
      throw std::runtime_error("Inconsistent receive buffer size.");

    // Send and receive data
    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_recv_fwd.data(),
                            _displs_recv_fwd.data(), data_type,
                            recv_buffer.data(), _sizes_send_fwd.data(),
                            displs_send_fwd.data(), data_type,
                            _comm_ghost_to_owner.comm(), &request);
  }

  /// Complete a non-blocking send of ghost values to the owning rank.
  /// This function complete the communication started by
  /// IndexMap::scatter_rev_begin.
  ///
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  void scatter_rev_end(MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();
    if (_displs_recv_fwd.size() == 1 and displs_send_fwd.size() == 1)
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  /// Send n values for each ghost index to owning to the process
  ///
  /// @param[in,out] local_data Local data associated with each owned
  /// local index to be sent to process where the data is ghosted. Size
  /// must be n * size_local().
  /// @param[in] remote_data Ghost data on this process received from
  /// the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  /// @param[in] op Sum or set received values in local_data
  template <typename T>
  void scatter_rev(xtl::span<T> local_data,
                   const xtl::span<const T>& remote_data, int n,
                   IndexMap::Mode op) const
  {
    MPI_Datatype data_type;
    if (n == 1)
      data_type = dolfinx::MPI::mpi_type<T>();
    else
    {
      MPI_Type_contiguous(n, dolfinx::MPI::mpi_type<T>(), &data_type);
      MPI_Type_commit(&data_type);
    }

    // Pack send buffer
    std::vector<T> buffer_send;
    buffer_send.resize(n * _displs_recv_fwd.back());
    for (std::size_t i = 0; i < _ghost_pos_recv_fwd.size(); ++i)
    {
      std::copy_n(std::next(remote_data.cbegin(), n * i), n,
                  std::next(buffer_send.begin(), n * _ghost_pos_recv_fwd[i]));
    }

    // Exchange data
    MPI_Request request;
    std::vector<T> buffer_recv(n * _shared_indices->array().size());
    scatter_rev_begin(xtl::span<const T>(buffer_send), data_type, request,
                      xtl::span<T>(buffer_recv));
    scatter_rev_end(request);

    // Copy or accumulate into "local_data"
    assert(local_data.size() == n * this->size_local());
    const std::vector<std::int32_t>& shared_indices = _shared_indices->array();
    switch (op)
    {
    case Mode::insert:
      for (std::size_t i = 0; i < shared_indices.size(); ++i)
      {
        std::copy_n(std::next(buffer_recv.cbegin(), n * i), n,
                    std::next(local_data.begin(), n * shared_indices[i]));
      }
      break;
    case Mode::add:
      for (std::size_t i = 0; i < shared_indices.size(); ++i)
      {
        for (int j = 0; j < n; ++j)
          local_data[shared_indices[i] * n + j] += buffer_recv[i * n + j];
      }
      break;
    }

    if (n != 1)
      MPI_Type_free(&data_type);
  }

private:
  // Range of indices (global) owned by this process
  std::array<std::int64_t, 2> _local_range;

  // Number indices across communicator
  std::int64_t _size_global;

  // MPI communicator (duplicated of 'input' communicator)
  dolfinx::MPI::Comm _comm;

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

  // MPI sizes and displacements for forward (owner -> ghost) scatter
  // Note: '_displs_send_fwd' can be got from _shared_indices->offsets()
  std::vector<std::int32_t> _sizes_send_fwd, _sizes_recv_fwd, _displs_recv_fwd;

  // Position in the recv buffer for a forward scatter for the ith ghost
  // index (_ghost[i]) entry
  std::vector<std::int32_t> _ghost_pos_recv_fwd;

  // Local-to-global map for ghost indices
  std::vector<std::int64_t> _ghosts;

  // List of owned local indices that are in the ghost (halo) region on
  // other ranks, grouped by rank in the neighbor communicator
  // (destination ranks in forward communicator and source ranks in the
  // reverse communicator), i.e. `_shared_indices.num_nodes() ==
  // size(_comm_owner_to_ghost)`. The array _shared_indices.offsets() is
  // equivalent to 'displs_send_fwd'.
  std::unique_ptr<graph::AdjacencyList<std::int32_t>> _shared_indices;
};

} // namespace dolfinx::common
