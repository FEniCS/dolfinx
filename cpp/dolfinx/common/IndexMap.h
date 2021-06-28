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

/// Compute layout data and ghost indices for a stacked (concatenated)
/// index map, i.e. 'splice' multiple maps into one. Communication is
/// required to compute the new ghost indices.
///
/// @param[in] maps List of (index map, block size) pairs
/// @returns The (0) global offset of a stacked map for this rank, (1)
///   local offset for each submap in the stacked map, and (2) new
///   indices for the ghosts for each submap (3) owner rank of each ghost
///   entry for each submap
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
  ///   of owned entries
  IndexMap(MPI_Comm comm, std::int32_t local_size);

  /// Create an index map with local_size owned indiced on this process
  ///
  /// @note Collective
  /// @param[in] mpi_comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMap, i.e. the number
  /// of owned entries
  /// @param[in] dest_ranks Ranks that 'ghost' indices that are owned by
  /// the calling rank. I.e., ranks that the caller will send data to
  /// when updating ghost values.
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] src_ranks Owner rank (on global communicator) of each
  /// entry in @p ghosts
  IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
           const xtl::span<const int>& dest_ranks,
           const xtl::span<const std::int64_t>& ghosts,
           const xtl::span<const int>& src_ranks);

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
  ///   this process, including ghosts
  std::vector<std::int64_t> global_indices() const;

  /// @todo Reconsider name
  /// Local (owned) indices shared with neighbor processes, i.e. are
  /// ghosts on other processes, grouped by sharing (neighbor)
  /// process(destination ranks in forward communicator and source ranks in the
  /// reverse communicator)
  /// @return List of indices that are ghosted on other processes
  const graph::AdjacencyList<std::int32_t>& shared_indices() const noexcept;

  /// Owner rank (on global communicator) of each ghost entry
  std::vector<int> ghost_owner_rank() const;

  /// @todo Aim to remove this function? If it's kept, should it work
  /// with neighborhood ranks?
  ///
  /// Compute map from each local (owned) index to the set of ranks that
  /// have the index as a ghost
  /// @return shared indices
  std::map<std::int32_t, std::set<int>> compute_shared_indices() const;

  /// Start a non-blocking send from the local owner of to process ranks
  /// that have the index as a ghost. The non-blocking communication is
  /// completed by calling IndexMap::scatter_fwd_end.
  ///
  /// @param[in] local_data Local data associated with each owned local
  /// index to be sent to process where the data is ghosted. Size must
  /// be `n * size_local()`, where `n` is the block size of the data to
  /// send.
  /// @param data_type The MPI data type. To send data with a block size
  /// use `MPI_Type_contiguous` with size `n`
  /// @param request The MPI request handle for tracking the status of
  /// the send
  /// @param send_buffer A buffer used to pack the send data. It must
  /// not be changed until after a call to IndexMap::scatter_fwd_end. It
  /// will be resized as required.
  /// @param recv_buffer  A buffer used for the received data. It must
  /// not be changed until after a call to IndexMap::scatter_fwd_end. It
  /// will be resized as required.
  template <typename T>
  void scatter_fwd_begin(const xtl::span<const T>& local_data,
                         MPI_Datatype& data_type, MPI_Request& request,
                         std::vector<T>& send_buffer,
                         std::vector<T>& recv_buffer) const
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
    if (static_cast<int>(local_data.size()) != n * size_local())
      throw std::runtime_error("Inconsistent data size.");

    // Copy data into send buffer
    send_buffer.resize(n * displs_send_fwd.back());
    const std::vector<std::int32_t>& indices = _shared_indices->array();
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
      std::copy_n(std::next(local_data.cbegin(), n * indices[i]), n,
                  std::next(send_buffer.begin(), n * i));
    }

    // Start send/receive
    recv_buffer.resize(n * _displs_recv_fwd.back());
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
  /// @param[in] remote_data The data array (ghost region) to fill with
  /// the received data
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  /// @param[in] recv_buffer The receive buffer. It must be the same as
  /// the buffer passed to IndexMap::scatter_fwd_begin.
  template <typename T>
  void scatter_fwd_end(const xtl::span<T>& remote_data, MPI_Request& request,
                       const xtl::span<const T>& recv_buffer) const
  {
    // Return early if there are no incoming or outgoing edges
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();
    if (_displs_recv_fwd.size() == 1 and displs_send_fwd.size() == 1)
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // Copy into ghost area ("remote_data")
    if (!remote_data.empty())
    {
      assert(remote_data.size() >= _ghosts.size());
      assert(remote_data.size() % _ghosts.size() == 0);
      const int n = remote_data.size() / _ghosts.size();
      std::vector<std::int32_t> displs = _displs_recv_fwd;
      for (std::size_t i = 0; i < _ghosts.size(); ++i)
      {
        const int p = _ghost_owners[i];
        std::copy_n(std::next(recv_buffer.cbegin(), n * displs[p]), n,
                    std::next(remote_data.begin(), n * i));
        displs[p] += 1;
      }
    }
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
      data_type = MPI::mpi_type<T>();
    else
    {
      MPI_Type_contiguous(n, dolfinx::MPI::mpi_type<T>(), &data_type);
      MPI_Type_commit(&data_type);
    }

    MPI_Request request;
    std::vector<T> buffer_send, buffer_recv;
    scatter_fwd_begin(local_data, data_type, request, buffer_send, buffer_recv);
    scatter_fwd_end(remote_data, request, xtl::span<const T>(buffer_recv));

    if (n != 1)
      MPI_Type_free(&data_type);
  }

  /// Start a non-blocking send of ghost values to the owning rank. The
  /// non-blocking communication is completed by calling
  /// IndexMap::scatter_rev_end.
  ///
  /// @param[in] remote_data Ghost data on this  process to be sent to
  /// the owner. Size must be `n * num_ghosts()`, where `n` is the block
  /// size of the data to send.
  /// @param data_type The MPI data type. To send data with a block size
  /// use `MPI_Type_contiguous` with size `n`
  /// @param request The MPI request handle for tracking the status of
  /// the send
  /// @param send_buffer A buffer used to pack the send data. It must
  /// not be changed until after a call to IndexMap::scatter_rev_end. It
  /// will be resized as required.
  /// @param recv_buffer  A buffer used for the received data. It must
  /// not be changed until after a call to IndexMap::scatter_rev_end. It
  /// will be resized as required.
  template <typename T>
  void scatter_rev_begin(const xtl::span<const T>& remote_data,
                         MPI_Datatype& data_type, MPI_Request& request,
                         std::vector<T>& send_buffer,
                         std::vector<T>& recv_buffer) const
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
    if (static_cast<int>(remote_data.size()) != n * _ghosts.size())
      throw std::runtime_error("Inconsistent data size.");

    // Pack send buffer
    send_buffer.resize(n * _displs_recv_fwd.back());
    std::vector<std::int32_t> displs(_displs_recv_fwd);
    for (std::size_t i = 0; i < _ghosts.size(); ++i)
    {
      const int p = _ghost_owners[i];
      std::copy_n(std::next(remote_data.cbegin(), n * i), n,
                  std::next(send_buffer.begin(), n * displs[p]));
      displs[p] += 1;
    }

    // Send and receive data
    recv_buffer.resize(n * displs_send_fwd.back());
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
  /// @param[in,out] local_data The data array (owned region) to sum/set
  /// with the received ghost data
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  /// @param[in] recv_buffer The receive buffer. It must be the same as
  /// the buffer passed to IndexMap::scatter_rev_begin.
  /// @param[in] op The accumulation options
  template <typename T>
  void scatter_rev_end(const xtl::span<T>& local_data, MPI_Request& request,
                       const xtl::span<const T>& recv_buffer,
                       IndexMap::Mode op) const
  {
    // Get displacement vector
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();

    // Return early if there are no incoming or outgoing edges
    if (_displs_recv_fwd.size() == 1 and displs_send_fwd.size() == 1)
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // Copy or accumulate into "local_data"
    if (std::int32_t size = this->size_local(); size > 0)
    {
      assert(local_data.size() >= size);
      assert(local_data.size() % size == 0);
      const int n = local_data.size() / size;
      const std::vector<std::int32_t>& shared_indices
          = _shared_indices->array();
      switch (op)
      {
      case Mode::insert:
        for (std::size_t i = 0; i < shared_indices.size(); ++i)
        {
          const std::int32_t index = shared_indices[i];
          std::copy_n(std::next(recv_buffer.cbegin(), n * i), n,
                      std::next(local_data.begin(), n * index));
        }
        break;
      case Mode::add:
        for (std::size_t i = 0; i < shared_indices.size(); ++i)
        {
          const std::int32_t index = shared_indices[i];
          for (int j = 0; j < n; ++j)
            local_data[index * n + j] += recv_buffer[i * n + j];
        }
        break;
      }
    }
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
      data_type = MPI::mpi_type<T>();
    else
    {
      MPI_Type_contiguous(n, dolfinx::MPI::mpi_type<T>(), &data_type);
      MPI_Type_commit(&data_type);
    }

    MPI_Request request;
    std::vector<T> buffer_send, buffer_recv;
    scatter_rev_begin(remote_data, data_type, request, buffer_send,
                      buffer_recv);
    scatter_rev_end(local_data, request, xtl::span<const T>(buffer_recv), op);

    if (n != 1)
      MPI_Type_free(&data_type);
  }

private:
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

  // MPI sizes and displacements for forward (owner -> ghost) scatter
  std::vector<int> _sizes_recv_fwd, _sizes_send_fwd, _displs_recv_fwd;

  // Local-to-global map for ghost indices
  std::vector<std::int64_t> _ghosts;

  // Owning neighborhood rank (out edge) on '_comm_owner_to_ghost'
  // communicator for each ghost index
  std::vector<std::int32_t> _ghost_owners;

  // List of owned local indices that are in the halo (ghost) region on
  // other ranks, grouped by rank in the neighbor communicator
  // (destination ranks in forward communicator and source ranks in the
  // reverse communicator), i.e. `_shared_indices.num_nodes() ==
  // size(_comm_owner_to_ghost)`.
  std::unique_ptr<graph::AdjacencyList<std::int32_t>> _shared_indices;
};

} // namespace dolfinx::common
