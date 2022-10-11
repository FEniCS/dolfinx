// Copyright (C) 2022 Igor Baratta and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPI.h"
#include <algorithm>
#include <mpi.h>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx::common
{
class IndexMap;

/// @brief A Scatterer supports the MPI scattering and gathering of data
/// that is associated with a common::IndexMap.
///
/// Scatter and gather operations uses MPI neighbourhood collectives.
/// The implementation is designed is for sparse communication patterns,
/// as it typical of patterns based on and IndexMap.
class Scatterer
{
public:
  /// @brief Create a scatterer
  /// @param[in] map The index map that describes the parallel layout of
  /// data
  /// @param[in] bs The block size of data associated with each index in
  /// `map` that will be scattered/gathered
  Scatterer(const IndexMap& map, int bs);

  /// @brief Start a non-blocking send of owned data to ranks that ghost
  /// the data.
  ///
  /// The communication is completed by calling
  /// Scatterer::scatter_fwd_end. The send and receive buffer should not
  /// be changed until after Scatterer::scatter_fwd_end has been called.
  ///
  /// @param[in] send_buffer Local data associated with each owned local
  /// index to be sent to process where the data is ghosted. It must not
  /// be changed until after a call to Scatterer::scatter_fwd_end. The
  /// order of data in the buffer is given by Scatterer::local_indices.
  /// @param recv_buffer A buffer used for the received data. The
  /// position of ghost entries in the buffer is given by
  /// Scatterer::remote_indices. The buffer must not be
  /// accessed or changed until after a call to
  /// Scatterer::scatter_fwd_end.
  /// @param request The MPI request handle for tracking the status of
  /// the non-blocking communication
  template <typename T>
  void scatter_fwd_begin(const std::span<const T>& send_buffer,
                         const std::span<T>& recv_buffer,
                         MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    int err = MPI_Ineighbor_alltoallv(
        send_buffer.data(), _sizes_local.data(), _displs_local.data(),
        MPI::mpi_type<T>(), recv_buffer.data(), _sizes_remote.data(),
        _displs_remote.data(), MPI::mpi_type<T>(), _comm0.comm(), &request);
    dolfinx::MPI::check_error(_comm0.comm(), err);
  }

  /// @brief Complete a non-blocking send from the local owner to
  /// process ranks that have the index as a ghost.
  ///
  /// This function completes the communication started by
  /// Scatterer::scatter_fwd_begin.
  ///
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  void scatter_fwd_end(MPI_Request& request) const;

  /// @brief Scatter data associated with owned indices to ghosting
  /// ranks.
  ///
  /// @note This function is intended for advanced usage, and in
  /// particular when using CUDA/device-aware MPI.
  ///
  /// @param[in] local_data All data associated with owned indices. Size
  /// is `size_local()` from the IndexMap used to create the scatterer,
  /// multiplied by the block size. The data for each index is blocked.
  /// @param[in] local_buffer Working buffer. The required size is given
  /// by Scatterer::local_buffer_size.
  /// @param[out] remote_buffer Working buffer. The required size is
  /// given by Scatterer::remote_buffer_size.
  /// @param[in] pack_fn Function to pack data from `local_data` into
  /// the send buffer. It is passed as an argument to support
  /// CUDA/device-aware MPI.
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  template <typename T, typename Functor>
  void scatter_fwd_begin(const std::span<const T>& local_data,
                         std::span<T> local_buffer, std::span<T> remote_buffer,
                         Functor pack_fn, MPI_Request& request) const
  {
    assert(local_buffer.size() == _local_inds.size());
    assert(remote_buffer.size() == _remote_inds.size());
    pack_fn(local_data, _local_inds, local_buffer);
    scatter_fwd_begin(std::span<const T>(local_buffer), remote_buffer, request);
  }

  /// @brief Complete a non-blocking send from the local owner to
  /// process ranks that have the index as a ghost, and unpack  received
  /// buffer into remote data.
  ///
  /// This function completes the communication started by
  /// Scatterer::scatter_fwd_begin.
  ///
  /// @param[in] remote_buffer Working buffer, same used in
  /// Scatterer::scatter_fwd_begin.
  /// @param[out] remote_data Received data associated with the ghost
  /// indices. The order follows the order of the ghost indices in the
  /// IndexMap used to create the scatterer. The size equal to the
  /// number of ghosts in the index map multiplied by the block size.
  /// The data for each index is blocked.
  /// @param[in] unpack_fn Function to unpack the received buffer into
  /// `remote_data`. It is passed as an argument to support
  /// CUDA/device-aware MPI.
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  template <typename T, typename Functor>
  void scatter_fwd_end(const std::span<const T>& remote_buffer,
                       std::span<T> remote_data, Functor unpack_fn,
                       MPI_Request& request) const
  {
    assert(remote_buffer.size() == _remote_inds.size());
    assert(remote_data.size() == _remote_inds.size());
    scatter_fwd_end(request);
    unpack_fn(remote_buffer, _remote_inds, remote_data,
              [](T /*a*/, T b) { return b; });
  }

  /// @brief Scatter data associated with owned indices to ghosting
  /// ranks.
  ///
  /// @param[in] local_data All data associated with owned indices. Size
  /// is `size_local()` from the IndexMap used to create the scatterer,
  /// multiplied by the block size. The data for each index is blocked
  /// @param[out] remote_data Received data associated with the ghost
  /// indices. The order follows the order of the ghost indices in the
  /// IndexMap used to create the scatterer. The size equal to the
  /// number of ghosts in the index map multiplied by the block size.
  /// The data for each index is blocked.
  template <typename T>
  void scatter_fwd(const std::span<const T>& local_data,
                   std::span<T> remote_data) const
  {
    MPI_Request request;
    std::vector<T> local_buffer(local_buffer_size(), 0);
    std::vector<T> remote_buffer(remote_buffer_size(), 0);
    auto pack_fn = [](const auto& in, const auto& idx, auto& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
    scatter_fwd_begin(local_data, std::span<T>(local_buffer),
                      std::span<T>(remote_buffer), pack_fn, request);

    auto unpack_fn = [](const auto& in, const auto& idx, auto& out, auto op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };

    scatter_fwd_end(std::span<const T>(remote_buffer), remote_data, unpack_fn,
                    request);
  }

  /// @brief Start a non-blocking send of ghost data to ranks that own
  /// the data.
  ///
  /// The communication is completed by calling
  /// Scatterer::scatter_rev_end. The send and receive buffers should not
  /// be changed until after Scatterer::scatter_rev_end has been called.
  ///
  /// @param[in] send_buffer Data associated with each ghost index. This
  /// data is sent to process that owns the index. It must not be
  /// changed until after a call to Scatterer::scatter_ref_end.
  /// @param recv_buffer Buffer used for the received data. The position
  /// of owned indices in the buffer is given by
  /// Scatterer::local_indices. Scatterer::local_displacements()[i] is
  /// the location of the first entry in `recv_buffer` received from
  /// neighbourhood rank i. The number of entries received from
  /// neighbourhood rank i is Scatterer::local_displacements()[i + 1] -
  /// Scatterer::local_displacements()[i]. `recv_buffer[j]` is the data
  /// associated with the index Scatterer::local_indices()[j] in the
  /// index map.
  ///
  /// The buffer must not be accessed or changed until after a call to
  /// Scatterer::scatter_fwd_end.
  /// @param request The MPI request handle for tracking the status of
  /// the non-blocking communication
  template <typename T>
  void scatter_rev_begin(const std::span<const T>& send_buffer,
                         const std::span<T>& recv_buffer,
                         MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Send and receive data
    int err = MPI_Ineighbor_alltoallv(
        send_buffer.data(), _sizes_remote.data(), _displs_remote.data(),
        MPI::mpi_type<T>(), recv_buffer.data(), _sizes_local.data(),
        _displs_local.data(), MPI::mpi_type<T>(), _comm1.comm(), &request);
    dolfinx::MPI::check_error(_comm1.comm(), err);
  }

  /// @brief End the reverse scatter communication.
  ///
  /// This function must be called after Scatterer::scatter_rev_begin.
  /// The buffers passed to Scatterer::scatter_rev_begin must not be
  /// modified until after the function has been called.
  ///
  /// @param[in] request The handle used when calling
  /// Scatterer::scatter_rev_begin
  void scatter_rev_end(MPI_Request& request) const;

  /// @brief Scatter data associated with ghost indices to owning ranks.
  ///
  /// @note This function is intended for advanced usage, and in
  /// particular when using CUDA/device-aware MPI.
  ///
  /// @tparam T The data type to send
  /// @tparam BinaryOp The reduction to perform when reducing data
  /// received from ghosting ranks to the value associated with the
  /// index on the owner
  /// @tparam Functor1 The pack function
  /// @tparam Functor2 The unpack function
  ///
  /// @param[in] remote_data Received data associated with the ghost
  /// indices. The order follows the order of the ghost indices in the
  /// IndexMap used to create the scatterer. The size equal to the
  /// number of ghosts in the index map multiplied by the block size.
  /// The data for each index is blocked.
  /// @param[out] local_buffer Working buffer. The requires size is given
  /// by Scatterer::local_buffer_size.
  /// @param[out] remote_buffer Working buffer. The requires size is
  /// given by Scatterer::remote_buffer_size.
  /// @param[in] pack_fn Function to pack data from `local_data` into
  /// the send buffer. It is passed as an argument to support
  /// CUDA/device-aware MPI.
  /// @param request The MPI request handle for tracking the status of
  /// the non-blocking communication
  template <typename T, typename Functor>
  void scatter_rev_begin(const std::span<const T>& remote_data,
                         std::span<T> remote_buffer, std::span<T> local_buffer,
                         Functor pack_fn, MPI_Request& request) const
  {
    assert(local_buffer.size() == _local_inds.size());
    assert(remote_buffer.size() == _remote_inds.size());
    pack_fn(remote_data, _remote_inds, remote_buffer);
    scatter_rev_begin(std::span<const T>(remote_buffer), local_buffer, request);
  }

  /// @brief End the reverse scatter communication, and unpack the received
  /// local buffer into local data.
  ///
  /// This function must be called after Scatterer::scatter_rev_begin.
  /// The buffers passed to Scatterer::scatter_rev_begin must not be
  /// modified until after the function has been called.
  /// @param[in] local_buffer Working buffer. Same buffer should be used in
  /// Scatterer::scatter_rev_begin.
  /// @param[out] local_data All data associated with owned indices.
  /// Size is `size_local()` from the IndexMap used to create the
  /// scatterer, multiplied by the block size. The data for each index
  /// is blocked.
  /// @param[in] unpack_fn Function to unpack the receive buffer into
  /// `local_data`. It is passed as an argument to support
  /// CUDA/device-aware MPI.
  /// @param[in] op The reduction operation when accumulating received
  /// values. To add the received values use `std::plus<T>()`.
  /// @param[in] request The handle used when calling
  /// Scatterer::scatter_rev_begin
  template <typename T, typename Functor, typename BinaryOp>
  void scatter_rev_end(const std::span<const T>& local_buffer,
                       std::span<T> local_data, Functor unpack_fn, BinaryOp op,
                       MPI_Request& request)
  {
    assert(local_buffer.size() == _local_inds.size());
    if (_local_inds.size() > 0)
      assert(*std::max_element(_local_inds.begin(), _local_inds.end())
             < std::int32_t(local_data.size()));
    scatter_rev_end(request);
    unpack_fn(local_buffer, _local_inds, local_data, op);
  }

  /// @brief Scatter data associated with ghost indices to ranks that
  /// own the indices.
  template <typename T, typename BinaryOp>
  void scatter_rev(std::span<T> local_data,
                   const std::span<const T>& remote_data, BinaryOp op)
  {
    std::vector<T> local_buffer(local_buffer_size(), 0);
    std::vector<T> remote_buffer(remote_buffer_size(), 0);
    auto pack_fn = [](const auto& in, const auto& idx, auto& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
    auto unpack_fn = [](const auto& in, const auto& idx, auto& out, auto op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };
    MPI_Request request;
    scatter_rev_begin(remote_data, std::span<T>(remote_buffer),
                      std::span<T>(local_buffer), pack_fn, request);
    scatter_rev_end(std::span<const T>(local_buffer), local_data, unpack_fn, op,
                    request);
  }

  /// @brief Size of buffer for local data (owned and shared) used in
  /// forward and reverse communication
  /// @return The required buffer size
  std::int32_t local_buffer_size() const noexcept;

  /// @brief Buffer size for remote data (ghosts) used in forward and
  /// reverse communication
  /// @return The required buffer size
  std::int32_t remote_buffer_size() const noexcept;

  /// Return a vector of local indices (owned) used to pack/unpack local data.
  /// These indices are grouped by neighbor process (process for which an index
  /// is a ghost).
  const std::vector<std::int32_t>& local_indices() const noexcept;

  /// Return a vector of remote indices (ghosts) used to pack/unpack ghost
  /// data. These indices are grouped by neighbor process (ghost owners).
  const std::vector<std::int32_t>& remote_indices() const noexcept;

  /// @brief The number values (block size) to send per index in the
  /// common::IndexMap use to create the scatterer
  /// @return The block size
  int bs() const noexcept;

private:
  // Block size
  int _bs;

  // Communicator where the source ranks own the indices in the callers
  // halo, and the destination ranks 'ghost' indices owned by the
  // caller. I.e.,
  // - in-edges (src) are from ranks that own my ghosts
  // - out-edges (dest) go to ranks that 'ghost' my owned indices
  dolfinx::MPI::Comm _comm0;

  // Communicator where the source ranks have ghost indices that are
  // owned by the caller, and the destination ranks are the owners of
  // indices in the callers halo region. I.e.,
  // - in-edges (src) are from ranks that 'ghost' my owned indices
  // - out-edges (dest) are to the owning ranks of my ghost indices
  dolfinx::MPI::Comm _comm1;

  // Permutation indices used to pack and unpack ghost data (remote)
  std::vector<std::int32_t> _remote_inds;

  // Number of remote indices (ghosts) for each neighbor process
  std::vector<int> _sizes_remote;

  // Displacements of remote data for mpi scatter and gather
  std::vector<int> _displs_remote;

  // Permutation indices used to pack and unpack local shared data
  // (owned indices that are shared with other processes). Indices are
  // grouped by neighbor process.
  std::vector<std::int32_t> _local_inds;

  // Number of local shared indices per neighbor process
  std::vector<int> _sizes_local;

  // Displacements of local data for mpi scatter and gather
  std::vector<int> _displs_local;
};
} // namespace dolfinx::common
