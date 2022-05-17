// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "IndexMap.h"
#include "sort.h>"
#include <algorithm>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <mpi.h>
#include <vector>

using namespace dolfinx;

namespace dolfinx::common
{

/// Manage MPI data scatter and gather
class Scatterer
{
public:
  /// @brief Create a scatter for MPI communication
  /// @param[in] map The index map the describes the parallel layout of
  /// data
  /// @param[in] bs The block size of data that will be communicated for
  /// indices in the `map`.
  Scatterer(const common::IndexMap& map, int bs)
      : _bs(bs), _comm_owner_to_ghost(MPI_COMM_NULL),
        _comm_ghost_to_owner(MPI_COMM_NULL)
  {
    if (map.overlapped())
    {
      // Get source (owner of ghosts) and destination (processes that
      // ghost an owned index) ranks
      const std::vector<int>& src_ranks = map.src();
      const std::vector<int>& dest_ranks = map.dest();

      // Check that assume src and dest ranks are unique and sorted
      assert(std::is_sorted(src_ranks.begin(), src_ranks.end()));
      assert(std::is_sorted(src_ranks.begin(), src_ranks.end()));

      // Create communicators with directed edges:
      // (0) owner -> ghost,
      // (1) ghost -> owner
      MPI_Comm comm0;
      MPI_Dist_graph_create_adjacent(
          map.comm(), src_ranks.size(), src_ranks.data(), MPI_UNWEIGHTED,
          dest_ranks.size(), dest_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
          false, &comm0);
      _comm_owner_to_ghost = dolfinx::MPI::Comm(comm0, false);

      MPI_Comm comm1;
      MPI_Dist_graph_create_adjacent(
          map.comm(), dest_ranks.size(), dest_ranks.data(), MPI_UNWEIGHTED,
          src_ranks.size(), src_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
          false, &comm1);
      _comm_ghost_to_owner = dolfinx::MPI::Comm(comm1, false);

      // Build permutation array that sorts ghost indices by owning rank
      const std::vector<int>& owners = map.owners();
      std::vector<std::int32_t> perm(owners.size());
      std::iota(perm.begin(), perm.end(), 0);
      dolfinx::argsort_radix<std::int32_t>(owners, perm);

      // Sort (I) ghost indices and (ii) ghost index owners by rank
      // (using perm array)
      const std::vector<std::int64_t>& ghosts = map.ghosts();
      std::vector<int> owners_sorted(owners.size());
      std::vector<std::int64_t> ghosts_sorted(owners.size());
      std::transform(perm.begin(), perm.end(), owners_sorted.begin(),
                     [&owners](auto idx) { return owners[idx]; });
      std::transform(perm.begin(), perm.end(), ghosts_sorted.begin(),
                     [&ghosts](auto idx) { return ghosts[idx]; });

      // Compute sizes and displacements of remote data (how many remote
      // to be sent/received grouped by neighbors)
      _sizes_remote.resize(src_ranks.size(), 0);
      _displs_remote.resize(src_ranks.size() + 1, 0);
      {
        auto begin = owners_sorted.begin();
        for (std::size_t i = 0; i < src_ranks.size(); i++)
        {
          auto upper
              = std::upper_bound(begin, owners_sorted.end(), src_ranks[i]);
          int num_ind = std::distance(begin, upper);
          _displs_remote[i + 1] = _displs_remote[i] + num_ind;
          _sizes_remote[i] = num_ind;
          begin = upper;
        }
      }

      // Compute sizes and displacements of local data (how many local
      // elements to be sent/received grouped by neighbors)
      _sizes_local.resize(dest_ranks.size());
      _displs_local.resize(_sizes_local.size() + 1);
      _sizes_remote.reserve(1);
      _sizes_local.reserve(1);
      MPI_Neighbor_alltoall(_sizes_remote.data(), 1,
                            MPI::mpi_type<std::int32_t>(), _sizes_local.data(),
                            1, MPI::mpi_type<std::int32_t>(),
                            _comm_ghost_to_owner.comm());

      std::partial_sum(_sizes_local.begin(), _sizes_local.end(),
                       std::next(_displs_local.begin()));

      // Send ghost indices (global) to owner, and receive owned indices
      // that are ghost in other process grouped by neighbor process.
      std::vector<std::int64_t> recv_buffer(_displs_local.back(), 0);
      assert((std::int32_t)ghosts_sorted.size() == _displs_remote.back());
      assert((std::int32_t)ghosts_sorted.size() == _displs_remote.back());
      MPI_Neighbor_alltoallv(
          ghosts_sorted.data(), _sizes_remote.data(), _displs_remote.data(),
          MPI_INT64_T, recv_buffer.data(), _sizes_local.data(),
          _displs_local.data(), MPI_INT64_T, _comm_ghost_to_owner.comm());

      std::array<std::int64_t, 2> range = map.local_range();
#ifndef NDEBUG
      // Check if all indices received are within the owned range.
      std::for_each(recv_buffer.begin(), recv_buffer.end(),
                    [range](auto idx)
                    { assert(idx >= range[0] and idx < range[1]); });
#endif

      // Scale sizes and displacements by block size
      {
        auto rescale = [](auto& x, int bs)
        {
          std::transform(x.begin(), x.end(), x.begin(),
                         [bs](auto e) { return e *= bs; });
        };
        rescale(_sizes_local, bs);
        rescale(_displs_local, bs);
        rescale(_sizes_remote, bs);
        rescale(_displs_remote, bs);
      }

      // Expand local indices using block size and convert it from
      // global to local numbering
      _local_inds.resize(recv_buffer.size() * _bs);
      std::int64_t offset = range[0] * _bs;
      for (std::size_t i = 0; i < recv_buffer.size(); i++)
        for (int j = 0; j < _bs; j++)
          _local_inds[i * _bs + j] = (recv_buffer[i] * _bs + j) - offset;

      // Expand remote indices using block size
      _remote_inds.resize(perm.size() * _bs);
      for (std::size_t i = 0; i < perm.size(); i++)
      {
        for (int j = 0; j < _bs; j++)
          _remote_inds[i * _bs + j] = perm[i] * _bs + j;
      }
    }
  }

  /// Return a lambda expression for packing data to be used in
  /// scatter_fwd and scatter_rev'.
  static auto pack()
  {
    return [](const auto& in, const auto& idx, auto& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
  }

  /// Return a lambda expression for unpacking received data from
  /// neighbors using 'scatter_fwd' or 'scatter_rev'.
  static auto unpack()
  {
    return [](const auto& in, const auto& idx, auto& out, auto op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };
  }

  /// Start a non-blocking send of owned data to ranks that ghost the
  /// data. The communication is completed by calling
  /// Scatterer::scatter_fwd_end. The send and receive buffer should not
  /// be changed until after Scatterer::scatter_fwd_end has been called.
  ///
  /// @param[in] send_buffer Local data associated with each owned local
  /// index to be sent to process where the data is ghosted. It must not
  /// be changed until after a call to IScatterer::scatter_fwd_end. The
  /// order of data in the buffer is given by Scatterer::local_indices.
  /// @param recv_buffer A buffer used for the received data. The
  /// position of ghost entries in the buffer is given by
  /// Scatterer::remote_indices. The buffer must not be
  /// accessed or changed until after a call to
  /// Scatterer::scatter_fwd_end.
  /// @param request The MPI request handle for tracking the status of
  /// the non-blocking communication
  template <typename T>
  void scatter_fwd_begin(const xtl::span<const T>& send_buffer,
                         const xtl::span<T>& recv_buffer,
                         MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_local.data(),
                            _displs_local.data(), MPI::mpi_type<T>(),
                            recv_buffer.data(), _sizes_remote.data(),
                            _displs_remote.data(), MPI::mpi_type<T>(),
                            _comm_owner_to_ghost.comm(), &request);
  }

  /// Complete a non-blocking send from the local owner to process
  /// ranks that have the index as a ghost. This function completes the
  /// communication started by Scatterer::scatter_fwd_begin.
  ///
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  void scatter_fwd_end(MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  /// TODO: Add documentation
  // NOTE: This function is not MPI-X friendly
  template <typename T, typename Functor1, typename Functor2>
  void scatter_fwd(const xtl::span<const T>& local_data,
                   xtl::span<T> remote_data, xtl::span<T> local_buffer,
                   xtl::span<T> remote_buffer, Functor1 pack_fn,
                   Functor2 unpack_fn) const
  {
    assert(local_buffer.size() == _local_inds.size());
    assert(remote_buffer.size() == _remote_inds.size());

    pack_fn(local_data, _local_inds, local_buffer);

    MPI_Request request;
    scatter_fwd_begin(xtl::span<const T>(local_buffer), remote_buffer, request);
    scatter_fwd_end(request);

    // Insert op
    auto op = [](T /*a*/, T b) { return b; };
    unpack_fn(remote_buffer, _remote_inds, remote_data, op);
  }

  /// TODO: Add documentation
  template <typename T>
  void scatter_fwd(const xtl::span<const T>& local_data,
                   xtl::span<T> remote_data) const
  {
    std::vector<T> local_buffer(local_buffer_size(), 0);
    std::vector<T> remote_buffer(remote_buffer_size(), 0);
    auto pack_fn = Scatterer::pack();
    auto unpack_fn = Scatterer::unpack();
    scatter_fwd(local_data, remote_data, xtl::span<T>(local_buffer),
                xtl::span<T>(remote_buffer), pack_fn, unpack_fn);
  }

  /// Start a non-blocking send of ghost values to the owning rank.
  template <typename T>
  void scatter_rev_begin(const xtl::span<const T>& send_buffer,
                         const xtl::span<T>& recv_buffer,
                         MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Send and receive data
    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_remote.data(),
                            _displs_remote.data(), MPI::mpi_type<T>(),
                            recv_buffer.data(), _sizes_local.data(),
                            _displs_local.data(), MPI::mpi_type<T>(),
                            _comm_ghost_to_owner.comm(), &request);
  }

  /// @brief End the reverse scatter communication
  /// @param[in] The request handle used when calling scatter_rev_begin
  void scatter_rev_end(MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  /// TODO
  template <typename T, typename BinaryOp, typename Functor1, typename Functor2>
  void scatter_rev(xtl::span<T> local_data,
                   const xtl::span<const T>& remote_data,
                   xtl::span<T> local_buffer, xtl::span<T> remote_buffer,
                   BinaryOp op, Functor1 pack_fn, Functor2 unpack_fn) const
  {
    assert(local_buffer.size() == _local_inds.size());
    assert(remote_buffer.size() == _remote_inds.size());

    // Pack send buffer
    pack_fn(remote_data, _remote_inds, remote_buffer);

    // Exchange data
    MPI_Request request;
    scatter_rev_begin(xtl::span<const T>(remote_buffer), local_buffer, request);
    scatter_rev_end(request);

    // Copy/accumulate into 'local_data'
    unpack_fn(local_buffer, _local_inds, local_data, op);
  }

  /// TODO
  template <typename T, typename BinaryOp>
  void scatter_rev(xtl::span<T> local_data,
                   const xtl::span<const T>& remote_data, BinaryOp op)
  {
    std::vector<T> local_buffer(local_buffer_size(), 0);
    std::vector<T> remote_buffer(remote_buffer_size(), 0);
    auto pack_fn = Scatterer::pack();
    auto unpack_fn = Scatterer::unpack();
    scatter_rev(local_data, remote_data, xtl::span<T>(local_buffer),
                xtl::span<T>(remote_buffer), op, pack_fn, unpack_fn);
  }

  /// @brief Size of buffer for local data (owned and shared) used in
  /// forward and reverse communication
  /// @return The required buffer size
  std::int32_t local_buffer_size() const noexcept
  {
    return _local_inds.size();
  };

  /// @brief Buffer size for remote data (ghosts) used in forward and
  /// reverse communication
  /// @return The required buffer size
  std::int32_t remote_buffer_size() const noexcept
  {
    return _remote_inds.size();
  };

  /// Return a vector of local indices (owned) used to pack/unpack local data.
  /// These indices are grouped by neighbor process (process for which an index
  /// is a ghost).
  const std::vector<std::int32_t>& local_indices() const noexcept
  {
    return _local_inds;
  }

  /// Return a vector of remote indices (ghosts) used to pack/unpack ghost
  /// data. These indices are grouped by neighbor process (ghost owners).
  const std::vector<std::int32_t>& remote_indices() const noexcept
  {
    return _remote_inds;
  }

  /// @brief The number values (block size) to send per index in the
  /// ::IndexMap use to create the scatterer
  /// @return The block size
  int bs() const noexcept { return _bs; }

private:
  // Block size
  int _bs;

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

  // Permutation indices used to pack and unpack ghost data (remote)
  std::vector<std::int32_t> _remote_inds;

  // Number of remote indices (ghosts) for each neighbor process
  std::vector<std::int32_t> _sizes_remote;

  // Displacements of remote data for mpi scatter and gather
  std::vector<std::int32_t> _displs_remote;

  // Permutation indices used to pack and unpack local shared data
  // (owned indices that are shared with other processes). Indices are
  // grouped by neighbor process.
  std::vector<std::int32_t> _local_inds;

  // Number of local shared indices per neighbor process
  std::vector<std::int32_t> _sizes_local;

  // Displacements of local data for mpi scatter and gather
  std::vector<std::int32_t> _displs_local;
};
} // namespace dolfinx::common