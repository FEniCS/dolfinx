// Copyright (C) 2022-2026 Igor Baratta, Garth N. Wells and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "IndexMap.h"
#include "MPI.h"
#include "ScatterPattern.h"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace dolfinx::common
{
namespace impl
{
/// @brief Scale MPI counts or displacements by a block size.
///
/// @param[in] x Counts or displacements for a block size of one.
/// @param[in] bs Block size.
/// @return Scaled counts or displacements.
inline std::vector<int> scale(std::span<const int> x, int bs)
{
  std::vector<int> y(x.size());
  std::ranges::transform(x, y.begin(), [bs](int e) { return e * bs; });
  return y;
}

/// @brief Expand indices by a block size, i.e. index `i` becomes the
/// `bs` indices `[i * bs, (i + 1) * bs)`.
///
/// The expansion is computed in `V`, so a `std::int64_t` expansion of a
/// large index does not overflow before it is widened.
///
/// @tparam V Value type of the expanded indices.
/// @param[in] indices Indices to expand.
/// @param[in] bs Block size, greater than zero.
/// @return Expanded indices.
template <std::integral V>
std::vector<V> expand_indices(std::span<const std::int32_t> indices, int bs)
{
  std::vector<V> idx(indices.size() * bs);
  for (std::size_t i = 0; i < indices.size(); i++)
  {
    const V base = static_cast<V>(indices[i]) * bs;
    for (int j = 0; j < bs; j++)
      idx[i * bs + j] = base + j;
  }
  return idx;
}
} // namespace impl

/// @brief A Scatterer supports the scattering and gathering of
/// distributed data that is associated with a common::IndexMap, using
/// MPI.
///
/// Scatter and gather operations can use:
/// 1. MPI neighbourhood collectives (recommended), or
/// 2. Non-blocking point-to-point communication modes.
///
/// The implementation is designed for sparse communication
/// patterns, as is typical of patterns based on an IndexMap.
///
/// A Scatterer is stateless, i.e. it provides the required information
/// and static data for a given parallel communication pattern but does
/// not provide any communication caches or track the status of MPI
/// requests. Callers of the a Scatterer's members are responsible for
/// managing buffer and MPI request handles.
///
/// A Scatterer is a block size-specific view onto a ScatterPattern that
/// is owned by the IndexMap. Scatterers built from one IndexMap share
/// that pattern, and with it the neighbourhood communicators, whatever
/// their block size or index container type.
///
/// @note Scatters may overlap: each caller supplies its own buffers and
/// MPI request, and forward and reverse scatters use separate
/// communicators. Because scatterers over one IndexMap share those
/// communicators, concurrent scatters in the same direction are matched
/// in the order they are started, so every rank must start them in the
/// same order. Starting them in a rank-dependent order silently
/// delivers one scatter's data to another. A caller that cannot meet
/// that requirement can opt out by constructing its own ScatterPattern
/// from the IndexMap and passing it here, giving it a private pair of
/// communicators.
///
/// @tparam Container Container type for storing the 'local' and
/// 'remote' indices. On CPUs this is normally
/// `std::vector<std::int32_t>`. For GPUs the container should store the
/// indices on the device, e.g. using
/// `thrust::device_vector<std::int32_t>`.
template <class Container = std::vector<std::int32_t>>
class Scatterer
{
  static_assert(std::is_integral_v<typename Container::value_type>);

  template <class>
  friend class Scatterer;

public:
  /// Container type used to store local and remote indices.
  using container_type = Container;

  /// @brief Create a scatterer for data with a layout described by a
  /// communication pattern and a block size.
  ///
  /// No MPI communication is performed; the pattern holds everything
  /// that requires it.
  ///
  /// @param[in] pattern Communication pattern of the index map that
  /// describes the parallel layout of the data. Must not be null.
  /// @param[in] bs Number of values associated with each index map
  /// index (the block size). Must be greater than zero.
  Scatterer(std::shared_ptr<const ScatterPattern> pattern, int bs)
      : _pattern(std::move(pattern))
  {
    if (!_pattern)
      throw std::invalid_argument("Scatterer requires a communication pattern");
    if (bs < 1)
    {
      throw std::invalid_argument("Scatterer block size must be greater than "
                                  "zero");
    }

    _sizes_remote = impl::scale(_pattern->sizes_remote(), bs);
    _displs_remote = impl::scale(_pattern->displs_remote(), bs);
    _sizes_local = impl::scale(_pattern->sizes_local(), bs);
    _displs_local = impl::scale(_pattern->displs_local(), bs);

    using V = typename container_type::value_type;
    _local_inds = impl::expand_indices<V>(_pattern->local_indices(), bs);
    _remote_inds = impl::expand_indices<V>(_pattern->perm(), bs);
  }

  /// @brief Create a scatterer for data with a layout described by an
  /// IndexMap and a block size.
  ///
  /// @note Collective on `map.comm()` if `map` has not yet built its
  /// communication pattern. See IndexMap::scatter_pattern.
  ///
  /// @param[in] map Index map that describes the parallel layout of
  /// data.
  /// @param[in] bs Number of values associated with each `map` index
  /// (the block size).
  Scatterer(const IndexMap& map, int bs) : Scatterer(map.scatter_pattern(), bs)
  {
  }

  /// Copy constructor
  Scatterer(const Scatterer& scatterer) = default;

  /// Move constructor
  Scatterer(Scatterer&& scatterer) = default;

  /// Destructor
  ~Scatterer() = default;

  /// Copy assignment
  Scatterer& operator=(const Scatterer& scatterer) = default;

  /// Move assignment
  Scatterer& operator=(Scatterer&& scatterer) = default;

  /// @brief Cast-copy constructor.
  ///
  /// Create a copy of a Scatterer, were the copy uses a different
  /// storage container for indices that are used in MPI communication.
  /// Example usage includes creating from a CPU-suitable Scatterer a
  /// GPU-suitable Scatterer that can be used with GPU-aware MPI to move
  /// data between devices. This would be typical when copying a
  /// la::Vector or la::MatrixCSR to/from a GPU. When copying a vector
  /// or matrix to/from a GPU, the underlying Scatter that manages
  /// parallel communication will usually be copied too with a different
  /// storage container.
  ///
  /// @param s Scatterer to copy
  template <class U>
  Scatterer(const Scatterer<U>& s)
      : _pattern(s._pattern),
        _remote_inds(s._remote_inds.begin(), s._remote_inds.end()),
        _sizes_remote(s._sizes_remote), _displs_remote(s._displs_remote),
        _local_inds(s._local_inds.begin(), s._local_inds.end()),
        _sizes_local(s._sizes_local), _displs_local(s._displs_local)
  {
  }

  /// @brief Start a non-blocking send of owned data to ranks that ghost
  /// the data using *MPI neighbourhood collective communication*
  /// (recommended).
  ///
  /// The communication is completed by calling Scatterer::scatter_end.
  /// See ::local_indices for instructions on packing `send_buffer` and
  /// ::remote_indices for instructions on unpacking `recv_buffer`.
  ///
  /// @note The send and receive buffers must **not** to be changed or
  /// accessed until after a call to Scatterer::scatter_end.
  ///
  /// @note The pointers `send_buffer` and `recv_buffer` must be
  /// pointers to the data on the *target device*. E.g., if the send and
  /// receive buffers are allocated on a GPU, the `send_buffer` and
  /// `recv_buffer` should be device pointers.
  ///
  /// @param[in] send_buffer Packed local data associated with each
  /// owned local index to be sent to processes where the data is
  /// ghosted. See Scatterer::local_indices for the order of the buffer
  /// and how to pack.
  /// @param[in,out] recv_buffer Buffer for storing received data. See
  /// Scatterer::remote_indices for the order of the buffer and how to unpack.
  /// @param[in] request MPI request handle for tracking the status of
  /// the non-blocking communication. The same request handle should be
  /// passed to Scatterer::scatter_end to complete the communication.
  template <typename T>
  void scatter_fwd_begin(const T* send_buffer, T* recv_buffer,
                         MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    int ierr = MPI_Ineighbor_alltoallv(
        send_buffer, _sizes_local.data(), _displs_local.data(),
        dolfinx::MPI::mpi_t<T>, recv_buffer, _sizes_remote.data(),
        _displs_remote.data(), dolfinx::MPI::mpi_t<T>, _pattern->comm0(),
        &request);
    dolfinx::MPI::check_error(_pattern->comm0(), ierr);
  }

  /// @brief Start a non-blocking send of owned data to ranks that ghost
  /// the data using *point-to-point MPI communication*.
  ///
  /// See ::scatter_fwd_begin for a detailed explanation of usage,
  /// including on the send and receive buffer packing and unpacking
  ///
  /// @note Use of the neighbourhood version of ::scatter_fwd_begin is
  /// recommended over this version.
  ///
  /// @param[in] send_buffer Send buffer.
  /// @param[in,out] recv_buffer Receive buffer.
  /// @param[in] requests List of MPI request handles. The length of the
  /// list must be ::num_p2p_requests()
  template <typename T>
  void scatter_fwd_begin(const T* send_buffer, T* recv_buffer,
                         std::span<MPI_Request> requests) const
  {
    std::span<const int> src = _pattern->src();
    std::span<const int> dest = _pattern->dest();
    if (requests.size() != dest.size() + src.size())
    {
      throw std::runtime_error(
          "Point-to-point scatterer has wrong number of MPI_Requests.");
    }

    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    MPI_Comm comm = _pattern->comm0();
    for (std::size_t i = 0; i < src.size(); ++i)
    {
      int ierr = MPI_Irecv(recv_buffer + _displs_remote[i], _sizes_remote[i],
                           dolfinx::MPI::mpi_t<T>, src[i], MPI_ANY_TAG, comm,
                           &requests[i]);
      dolfinx::MPI::check_error(comm, ierr);
    }

    for (std::size_t i = 0; i < dest.size(); ++i)
    {
      int ierr = MPI_Isend(send_buffer + _displs_local[i], _sizes_local[i],
                           dolfinx::MPI::mpi_t<T>, dest[i], 0, comm,
                           &requests[i + src.size()]);
      dolfinx::MPI::check_error(comm, ierr);
    }
  }

  /// @brief Start a non-blocking send of ghost data to ranks that own
  /// the data using *MPI neighbourhood collective communication*
  /// (recommended).
  ///
  /// The communication is completed by calling Scatterer::scatter_end.
  /// See ::remote_indices for instructions on packing `send_buffer` and
  /// ::local_indices  for instructions on unpacking `recv_buffer`.
  ///
  /// @note The send and receive buffers must **not** to be changed or
  /// accessed until after a call to Scatterer::scatter_end.
  ///
  /// @note The pointers `send_buffer` and `recv_buffer` must be
  /// pointers to the data on the *target device*. E.g., if the send and
  /// receive buffers are allocated on a GPU, the `send_buffer` and
  /// `recv_buffer` should be device pointers.
  ///
  /// @param[in] send_buffer Data associated with each ghost index. This
  /// data is sent to process that owns the index. See
  /// Scatterer::remote_indices for the order of the buffer and how to
  /// pack.
  /// @param[in,out] recv_buffer Buffer for storing received data. See
  /// Scatterer::local_indices for the order of the buffer and how to
  /// unpack.
  /// @param[in] request MPI request handle for tracking the status of
  /// the non-blocking communication. The same request handle should be
  /// passed to Scatterer::scatter_end to complete the communication.
  template <typename T>
  void scatter_rev_begin(const T* send_buffer, T* recv_buffer,
                         MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    int ierr = MPI_Ineighbor_alltoallv(
        send_buffer, _sizes_remote.data(), _displs_remote.data(),
        dolfinx::MPI::mpi_t<T>, recv_buffer, _sizes_local.data(),
        _displs_local.data(), dolfinx::MPI::mpi_t<T>, _pattern->comm1(),
        &request);
    dolfinx::MPI::check_error(_pattern->comm1(), ierr);
  }

  /// @brief Start a non-blocking send of ghost data to ranks that own
  /// the data using *point-to-point MPI communication*.
  ///
  /// See ::scatter_rev_begin for a detailed explanation of usage,
  /// including on the send and receive buffer packing and unpacking
  ///
  /// @note Use of the neighbourhood version of ::scatter_rev_begin is
  /// recommended over this version
  ///
  /// @param[in] send_buffer Send buffer.
  /// @param[in,out] recv_buffer Receive buffer.
  /// @param[in] requests List of MPI request handles. The length of the
  /// list must be ::num_p2p_requests()
  template <typename T>
  void scatter_rev_begin(const T* send_buffer, T* recv_buffer,
                         std::span<MPI_Request> requests) const
  {
    std::span<const int> src = _pattern->src();
    std::span<const int> dest = _pattern->dest();
    if (requests.size() != dest.size() + src.size())
    {
      throw std::runtime_error(
          "Point-to-point scatterer has wrong number of MPI_Requests.");
    }

    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Start non-blocking send from this process to ghost owners
    MPI_Comm comm = _pattern->comm0();
    for (std::size_t i = 0; i < dest.size(); i++)
    {
      int ierr = MPI_Irecv(recv_buffer + _displs_local[i], _sizes_local[i],
                           dolfinx::MPI::mpi_t<T>, dest[i], MPI_ANY_TAG, comm,
                           &requests[i]);
      dolfinx::MPI::check_error(comm, ierr);
    }

    // Start non-blocking receive from neighbor process for which an
    // owned index is a ghost
    for (std::size_t i = 0; i < src.size(); i++)
    {
      int ierr = MPI_Isend(send_buffer + _displs_remote[i], _sizes_remote[i],
                           dolfinx::MPI::mpi_t<T>, src[i], 0, comm,
                           &requests[i + dest.size()]);
      dolfinx::MPI::check_error(comm, ierr);
    }
  }

  /// @brief Complete non-blocking MPI point-to-point sends.
  ///
  /// This function completes the communication started by
  /// ::scatter_fwd_begin or ::scatter_rev_begin.
  ///
  /// @param[in] requests MPI request handles for tracking the status of
  /// sends.
  void scatter_end(std::span<MPI_Request> requests) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Wait for communication to complete
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  }

  /// @brief Complete a non-blocking MPI neighbourhood collective send.
  ///
  /// This function completes the communication started by
  /// ::scatter_fwd_begin or ::scatter_rev_begin.
  ///
  /// @param[in] request MPI request handle for tracking the status of
  /// the send.
  void scatter_end(MPI_Request& request) const
  {
    scatter_end(std::span<MPI_Request>(&request, 1));
  }

  /// @brief Array of indices for packing/unpacking owned data to/from a
  /// send/receive buffer.
  ///
  /// For a forward scatter, the indices are used to copy required
  /// entries in the owned part of the data array into the appropriate
  /// position in a send buffer. For a reverse scatter, indices are used
  /// for assigning (accumulating) the receive buffer values into
  /// correct position in the owned part of the data array.
  ///
  /// For a forward scatter, if `x` is the owned part of an array and
  /// `send_buffer` is the send buffer, `send_buffer` is packed such
  /// that:
  ///
  ///     auto& idx = scatterer.local_indices()
  ///     std::vector<T> send_buffer(idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         send_buffer[i] = x[idx[i]];
  ///
  /// For a reverse scatter, if `recv_buffer` is the received buffer,
  /// then `x` is updated by
  ///
  ///     auto& idx = scatterer.local_indices()
  ///     std::vector<T> recv_buffer(idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         x[idx[i]] = op(recv_buffer[i], x[idx[i]]);
  ///
  /// where `op` is a binary operation, e.g. `x[idx[i]] = buffer[i]` or
  /// `x[idx[i]] += buffer[i]`.
  ///
  /// @return Indices container.
  const container_type& local_indices() const noexcept { return _local_inds; }

  /// @brief Array of indices for packing/unpacking ghost data to/from a
  /// send/receive buffer.
  ///
  /// For a forward scatter, the indices are to copy required entries in
  /// the owned array into the appropriate position in a send buffer.
  /// For a reverse scatter, indices are used for assigning
  /// (accumulating) the receive buffer values to correct position in
  /// the owned array.
  ///
  /// For a forward scatter, if `xg` is the ghost part of the data array
  /// and `recv_buffer` is the receive buffer, `xg` is updated that
  ///
  ///     auto& idx = scatterer.remote_indices()
  ///     std::vector<T> recv_buffer(idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         xg[idx[i]] = recv_buffer[i];
  ///
  /// For a reverse scatter, if `send_buffer` is the send buffer, then
  /// `send_buffer` is packaged such that:
  ///
  ///     auto& idx = scatterer.remote_indices()
  ///     std::vector<T> send_buffer(idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         send_buffer[i] = xg[idx[i]];
  ///
  /// @return Indices container.
  const container_type& remote_indices() const noexcept { return _remote_inds; }

  /// @brief Number of required `MPI_Request`s for point-to-point
  /// communication.
  ///
  /// @return Number of required MPI request handles.
  std::size_t num_p2p_requests() const noexcept
  {
    return _pattern->dest().size() + _pattern->src().size();
  }

private:
  // Block size-independent communication pattern, shared with every
  // other Scatterer built from the same IndexMap
  std::shared_ptr<const ScatterPattern> _pattern;

  // Permutation indices used to pack and unpack ghost data (remote)
  container_type _remote_inds;

  // Number of remote indices (ghosts) for each neighbor process
  std::vector<int> _sizes_remote;

  // Displacements of remote data for mpi scatter and gather
  std::vector<int> _displs_remote;

  // Permutation indices used to pack and unpack local shared data
  // (owned indices that are shared with other processes). Indices are
  // grouped by neighbor process.
  container_type _local_inds;

  // Number of local shared indices per neighbor process
  std::vector<int> _sizes_local;

  // Displacements of local data for mpi scatter and gather
  std::vector<int> _displs_local;
};
} // namespace dolfinx::common
