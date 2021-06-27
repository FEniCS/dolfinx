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
    reverse,  // Ghost to owner
    forward,  // Owner to ghost
    symmetric // Symmetric. NOTE: To be removed
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
  /// @param[in] dir Edge direction of communicator (forward, reverse,
  /// symmetric)
  /// @return A neighborhood communicator for the specified edge direction
  MPI_Comm comm(Direction dir = Direction::symmetric) const;

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

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as n * size_local().
  ///
  /// @param[in] local_data Local data associated with each owned local
  /// index to be sent to process where the data is ghosted. Size must
  /// be n * size_local().
  /// @param data_type
  /// @param request
  /// @param send_buffer
  /// @param recv_buffer
  template <typename T>
  void scatter_fwd_begin(const xtl::span<const T>& local_data,
                         MPI_Datatype& data_type, MPI_Request& request,
                         std::vector<T>& send_buffer,
                         std::vector<T>& recv_buffer) const
  {
    // Send displacements
    const std::vector<int32_t>& displs_send = _shared_indices->offsets();

    // if (_displs_recv_fwd.size() == 1 and displs_send.size() == 1)
    //   return;

    // std::cout << "Scatter fwd  0" << std::endl;
    if (static_cast<std::int32_t>(local_data.size()) < size_local()
        and local_data.size() % size_local() != 0)
    {
      throw std::runtime_error("Invalid remote size in scatter_fwd");
    }

    std::cout << "Scatter fwd  1" << std::endl;
    // #ifdef DEBUG
    int n;
    MPI_Type_size(data_type, &n);
    n /= sizeof(T);
    if (size_local() > 0
        and n != static_cast<int>(local_data.size() / size_local()))
    {

      throw std::runtime_error("Inconsistent block size.");
    }

    std::cout << "Scatter fwd  2" << std::endl;

    // Copy data into send buffer
    send_buffer.resize(n * displs_send.back() + 1); // Add '1' for OpenMPI bug
    const std::vector<std::int32_t>& indices = _shared_indices->array();
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
      const std::int32_t index = indices[i];
      for (int j = 0; j < n; ++j)
        send_buffer[i * n + j] = local_data[index * n + j];
    }

    std::cout << "Scatter fwd  3K: " << _displs_recv_fwd.back() << std::endl;

    // Start send/receive
    recv_buffer.resize(n * _displs_recv_fwd.back()
                       + 1); // Add '1' for OpenMPI bug
    std::cout << "Sizes: " << n << std::endl;
    std::cout << send_buffer.size() << std::endl;
    std::cout << _sizes_send_fwd.size() << std::endl; //
    std::cout << displs_send.size() << std::endl;
    std::cout << recv_buffer.size() << std::endl;     //
    std::cout << _sizes_recv_fwd.size() << std::endl; //
    std::cout << _displs_recv_fwd.size() << std::endl;
    std::cout << "End sizes: " << std::endl;
    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_send_fwd.data(),
                            displs_send.data(), data_type, recv_buffer.data(),
                            _sizes_recv_fwd.data(), _displs_recv_fwd.data(),
                            data_type, _comm_owner_to_ghost.comm(), &request);
    std::cout << "Scatter fwd  4" << std::endl;
  }

  /// Complete forward scatter
  template <typename T>
  void scatter_fwd_end(const xtl::span<T>& remote_data, MPI_Request& request,
                       const xtl::span<const T>& recv_buffer) const
  {
    // const std::vector<int32_t>& displs_send = _shared_indices->offsets();
    // if (_displs_recv_fwd.size() == 1 and displs_send.size() == 1)
    //   return;

    std::cout << "fwd_end 3xx" << std::endl;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    std::cout << "fwd_end 4xx" << std::endl;

    // Copy into ghost area ("remote_data")
    if (!remote_data.empty())
    {
      assert(remote_data.size() >= _ghosts.size());
      assert(remote_data.size() % _ghosts.size() == 0);
      const int n = remote_data.size() / _ghosts.size();
      std::vector<std::int32_t> displs = _displs_recv_fwd;
      for (std::size_t i = 0; i < _ghosts.size(); ++i)
      {
        const int np = _ghost_owners[i];
        for (int j = 0; j < n; ++j)
          remote_data[i * n + j] = recv_buffer[n * displs[np] + j];
        displs[np] += 1;
      }
    }
    std::cout << "fwd_end 5x" << std::endl;
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
    std::cout << "Scatter pre 0" << std::endl;
    MPI_Datatype data_type;
    if (n == 1)
      data_type = MPI::mpi_type<T>();
    else
    {
      MPI_Type_contiguous(n, dolfinx::MPI::mpi_type<T>(), &data_type);
      MPI_Type_commit(&data_type);
    }

    std::cout << "Scatter pre 1" << std::endl;
    MPI_Request request;
    std::vector<T> buffer_send, buffer_recv;
    scatter_fwd_begin(local_data, data_type, request, buffer_send, buffer_recv);
    std::cout << "Scatter pre 2n" << std::endl;
    scatter_fwd_end(remote_data, request, xtl::span<const T>(buffer_recv));

    std::cout << "Scatter pre 3" << std::endl;
    if (n != 1)
      MPI_Type_free(&data_type);
    std::cout << "Scatter pre 4" << std::endl;
  }

  /// Send n values for each ghost index to owning to the process
  ///
  /// @param[in,out] local_data Local data associated with each owned
  ///   local index to be sent to process where the data is ghosted.
  ///   Size must be n * size_local().
  /// @param[in] remote_data Ghost data on this process received from
  ///   the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  /// @param[in] op Sum or set received values in local_data
  template <typename T>
  void scatter_rev(xtl::span<T> local_data,
                   const xtl::span<const T>& remote_data, int n,
                   IndexMap::Mode op) const;

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

  // MPI sizes and displacements for ghost update/scatter
  std::vector<std::int32_t> _sizes_recv_fwd, _sizes_send_fwd, _displs_recv_fwd;

  // TODO: remove
  dolfinx::MPI::Comm _comm_symmetric;

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
