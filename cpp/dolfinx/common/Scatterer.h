// Copyright (C) 2022-2026 Igor Baratta and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "IndexMap.h"
#include "MPI.h"
#include "sort.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <mpi.h>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace dolfinx::common
{
/// @brief A Scatterer supports the scattering and gathering of
/// distributed data that is associated with a common::IndexMap, using
/// MPI.
///
/// Scatter and gather operations use MPI neighbourhood collectives.
///
/// The implementation is designed for sparse communication
/// patterns, as is typical of patterns based on an IndexMap.
///
/// A Scatterer is stateless, i.e. it provides the required information
/// and static data for a given parallel communication pattern but does
/// not provide any communication caches or track the status of MPI
/// requests. Callers of a Scatterer's members are responsible for
/// managing buffer and MPI request handles.
///
/// Creating, copying and destroying a Scatterer are collective, since
/// they create, duplicate and free MPI communicators. Move construction
/// is not collective, but move assignment is, since it frees the
/// communicators held by the assignment target.
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

  /// @brief Create a scatterer for data with a layout described by an
  /// IndexMap.
  ///
  /// @note Collective.
  ///
  /// @param[in] map Index map that describes the parallel layout of
  /// data.
  explicit Scatterer(const IndexMap& map)
      : _sizes_remote(map.src().size(), 0),
        _displs_remote(map.src().size() + 1), _sizes_local(map.dest().size()),
        _displs_local(map.dest().size() + 1)
  {
    if (dolfinx::MPI::size(map.comm()) == 1)
      return;

    int ierr;
    const std::span<const int> src = map.src();
    const std::span<const int> dest = map.dest();

    // Check that src and dest ranks are unique and sorted
    assert(std::ranges::is_sorted(src));
    assert(std::ranges::is_sorted(dest));

    // Create communicators with directed edges:
    // (0) owner -> ghost,
    // (1) ghost -> owner
    MPI_Comm comm0;
    ierr = MPI_Dist_graph_create_adjacent(
        map.comm(), src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
        dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    _comm0 = dolfinx::MPI::Comm(comm0, false);
    dolfinx::MPI::check_error(map.comm(), ierr);

    MPI_Comm comm1;
    ierr = MPI_Dist_graph_create_adjacent(
        map.comm(), dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(),
        src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);
    _comm1 = dolfinx::MPI::Comm(comm1, false);
    dolfinx::MPI::check_error(map.comm(), ierr);

    // Build permutation array that sorts ghost indices by owning rank
    std::span owners = map.owners();
    std::vector<std::int32_t> perm(owners.size());
    std::iota(perm.begin(), perm.end(), 0);
    dolfinx::radix_sort(perm, [&owners](auto index) { return owners[index]; });

    // Sort (i) ghost indices and (ii) ghost index owners by rank
    // (using perm array)
    std::span ghosts = map.ghosts();
    std::vector<int> owners_sorted(owners.size());
    std::vector<std::int64_t> ghosts_sorted(owners.size());
    std::ranges::transform(perm, owners_sorted.begin(),
                           [&owners](auto idx) { return owners[idx]; });
    std::ranges::transform(perm, ghosts_sorted.begin(),
                           [&ghosts](auto idx) { return ghosts[idx]; });

    // For data associated with ghost indices, packed by owning
    // (neighbourhood) rank, compute sizes and displacements. I.e., when
    // sending ghost index data from this rank to the owning ranks,
    // disp[i] is the first entry in the buffer sent to neighbourhood
    // rank i, and disp[i + 1] - disp[i] is the number of values sent to
    // rank i.
    assert(_sizes_remote.size() == src.size());
    assert(_displs_remote.size() == src.size() + 1);
    auto begin = owners_sorted.begin();
    for (std::size_t i = 0; i < src.size(); i++)
    {
      auto upper = std::ranges::upper_bound(begin, owners_sorted.end(), src[i]);
      std::size_t num_ind = std::ranges::distance(begin, upper);
      _displs_remote[i + 1] = _displs_remote[i] + num_ind;
      _sizes_remote[i] = num_ind;
      begin = upper;
    }

    // For data associated with owned indices that are ghosted by other
    // ranks, compute the size and displacement arrays. When sending
    // data associated with ghost indices to the owner, these size and
    // displacement arrays are for the receive buffer.

    // Compute sizes and displacements of local data (how many local
    // elements to be sent/received grouped by neighbors)
    assert(_sizes_local.size() == dest.size());
    assert(_displs_local.size() == dest.size() + 1);
    _sizes_remote.reserve(1); // ensure data is not a nullptr
    _sizes_local.reserve(1);  // ensure data is not a nullptr
    ierr
        = MPI_Neighbor_alltoall(_sizes_remote.data(), 1, MPI_INT,
                                _sizes_local.data(), 1, MPI_INT, _comm1.comm());
    dolfinx::MPI::check_error(_comm1.comm(), ierr);

    std::inclusive_scan(_sizes_local.begin(), _sizes_local.end(),
                        std::next(_displs_local.begin()));

    assert(static_cast<int>(ghosts_sorted.size()) == _displs_remote.back());

    // Send ghost global indices to owning rank, and receive owned
    // indices that are ghosts on other ranks
    std::vector<std::int64_t> recv_buffer(_displs_local.back(), 0);
    ierr = MPI_Neighbor_alltoallv(
        ghosts_sorted.data(), _sizes_remote.data(), _displs_remote.data(),
        MPI_INT64_T, recv_buffer.data(), _sizes_local.data(),
        _displs_local.data(), MPI_INT64_T, _comm1.comm());
    dolfinx::MPI::check_error(_comm1.comm(), ierr);

    const std::array<std::int64_t, 2> range = map.local_range();
#ifndef NDEBUG
    // Check that all received indices are within the owned range
    std::ranges::for_each(recv_buffer, [range](auto idx)
                          { assert(idx >= range[0] and idx < range[1]); });
#endif

    // Sizes, displacements and indices are all in blocks. The block
    // size enters only through the MPI datatype used to send them, and
    // through the caller's pack/unpack.
    {
      // Convert the received indices from global to local numbering
      std::vector<typename container_type::value_type> idx(recv_buffer.size());
      std::ranges::transform(recv_buffer, idx.begin(),
                             [offset = range[0]](auto i) ->
                             typename container_type::value_type
                             { return i - offset; });
      _local_inds = std::move(idx);
    }

    _remote_inds = container_type(perm.begin(), perm.end());
  }

  /// @brief Cast-copy constructor.
  ///
  /// Create a copy of a Scatterer, where the copy uses a different
  /// storage container for indices that are used in MPI communication.
  /// Example usage includes creating from a CPU-suitable Scatterer a
  /// GPU-suitable Scatterer that can be used with GPU-aware MPI to move
  /// data between devices. This would be typical when copying a
  /// la::Vector or la::MatrixCSR to/from a GPU. When copying a vector
  /// or matrix to/from a GPU, the underlying Scatter that manages
  /// parallel communication will usually be copied too with a different
  /// storage container.
  ///
  /// @note Collective. The neighbourhood communicators are duplicated,
  /// so all ranks must make the copy together.
  ///
  /// @param s Scatterer to copy
  template <class U>
  Scatterer(const Scatterer<U>& s)
      : _comm0(s._comm0), _comm1(s._comm1),
        _remote_inds(s._remote_inds.begin(), s._remote_inds.end()),
        _sizes_remote(s._sizes_remote), _displs_remote(s._displs_remote),
        _local_inds(s._local_inds.begin(), s._local_inds.end()),
        _sizes_local(s._sizes_local), _displs_local(s._displs_local)
  {
  }

  /// Copy constructor
  ///
  /// @note Collective, as for the cast-copy constructor. Move instead
  /// where the original is no longer required.
  Scatterer(const Scatterer& scatterer) = default;

  /// Move constructor
  ///
  /// @note Not collective, unlike the copy constructors: the
  /// communicators are taken over rather than duplicated.
  Scatterer(Scatterer&& scatterer) = default;

  /// Destructor
  ///
  /// @note Collective, since the communicators are freed.
  ~Scatterer() = default;

  // Copy assignment (deleted). dolfinx::MPI::Comm cannot be copied into
  // an existing object.
  Scatterer& operator=(const Scatterer& scatterer) = delete;

  /// Move assignment
  ///
  /// @note Collective if this Scatterer holds communicators, since
  /// assigning to it frees them.
  Scatterer& operator=(Scatterer&& scatterer) = default;

  /// @brief Start a non-blocking neighbourhood collective exchange of
  /// owned data with the ranks that ghost it.
  ///
  /// The communication is completed by calling
  /// Scatterer::scatter_fwd_end. See ::local_indices_block for how to
  /// pack `send_buffer` and ::remote_indices_block for how to unpack
  /// `recv_buffer`.
  ///
  /// This is a differently named function rather than an overload of
  /// ::scatter_fwd_begin because the underlying type of `MPI_Datatype`
  /// is implementation-defined, and is an integer type in some MPI
  /// implementations, so overloading on `int` and `MPI_Datatype` is not
  /// portably unambiguous.
  ///
  /// @note Collective MPI operation. Every rank in the communicator
  /// must call this function, including ranks without neighbours.
  ///
  /// @note The send and receive buffers must **not** be changed or
  /// accessed until after a call to Scatterer::scatter_fwd_end.
  ///
  /// @note The pointers `send_buffer` and `recv_buffer` must be
  /// pointers to the data on the *target device*. E.g., if the send and
  /// receive buffers are allocated on a GPU, the `send_buffer` and
  /// `recv_buffer` should be device pointers.
  ///
  /// @param[in] send_buffer Packed local data associated with each
  /// owned local index to be sent to processes where the data is
  /// ghosted. See Scatterer::local_indices_block for the order of the
  /// buffer and how to pack.
  /// @param[in,out] recv_buffer Buffer for storing received data. See
  /// Scatterer::remote_indices_block for the order of the buffer and
  /// how to unpack.
  /// @param[in] type MPI datatype for the data associated with one
  /// index, e.g. `dolfinx::MPI::Datatype<T>(bs).type()`. Buffer counts
  /// and displacements are in units of `type`, and the same type must
  /// be used on all ranks. MPI keeps a datatype alive until
  /// communication using it has completed, so `type` may be freed as
  /// soon as this function returns.
  /// @param[out] request Handle for tracking the status of the
  /// non-blocking communication. Any value passed in is overwritten,
  /// and `MPI_REQUEST_NULL` is returned when this rank has nothing to
  /// communicate. The same handle must be passed to
  /// Scatterer::scatter_fwd_end to complete the communication.
  template <typename T>
  void scatter_fwd_begin_dtype(const T* send_buffer, T* recv_buffer,
                               MPI_Datatype type, MPI_Request& request) const
  {
    if (!has_neighbours())
    {
      request = MPI_REQUEST_NULL;
      return;
    }

    int ierr = MPI_Ineighbor_alltoallv(
        send_buffer, _sizes_local.data(), _displs_local.data(), type,
        recv_buffer, _sizes_remote.data(), _displs_remote.data(), type,
        _comm0.comm(), &request);
    dolfinx::MPI::check_error(_comm0.comm(), ierr);
  }

  /// @brief Start a non-blocking neighbourhood collective exchange of
  /// owned data with the ranks that ghost it.
  ///
  /// As ::scatter_fwd_begin_dtype, but with the MPI datatype built from
  /// a block size.
  ///
  /// @param[in] send_buffer Packed local data associated with each
  /// owned local index to be sent to processes where the data is
  /// ghosted. See Scatterer::local_indices_block for the order of the
  /// buffer and how to pack.
  /// @param[in,out] recv_buffer Buffer for storing received data. See
  /// Scatterer::remote_indices_block for the order of the buffer and
  /// how to unpack.
  /// @param[in] bs Number of values per index map index (the block
  /// size). The buffers hold `bs` values for each index in
  /// ::local_indices_block and ::remote_indices_block respectively.
  /// @param[out] request Handle for tracking the status of the
  /// non-blocking communication. Any value passed in is overwritten,
  /// and `MPI_REQUEST_NULL` is returned when this rank has nothing to
  /// communicate. The same handle must be passed to
  /// Scatterer::scatter_fwd_end to complete the communication.
  template <typename T>
  void scatter_fwd_begin(const T* send_buffer, T* recv_buffer, int bs,
                         MPI_Request& request) const
  {
    // Checked here too, to avoid building a datatype that will not be
    // used
    if (!has_neighbours())
    {
      request = MPI_REQUEST_NULL;
      return;
    }

    dolfinx::MPI::Datatype<T> type(bs);
    scatter_fwd_begin_dtype(send_buffer, recv_buffer, type.type(), request);
  }

  /// @brief Complete a non-blocking MPI neighbourhood collective send.
  ///
  /// This function completes the communication started by
  /// ::scatter_fwd_begin or ::scatter_fwd_begin_dtype.
  ///
  /// @note Local completion of the caller's own request, not itself
  /// collective. Every rank that called ::scatter_fwd_begin must
  /// still call this before reusing the buffers.
  ///
  /// @param[in,out] request Handle returned by the matching begin
  /// call. Set to `MPI_REQUEST_NULL` once the communication has
  /// completed.
  void scatter_fwd_end(MPI_Request& request) const
  {
    if (!has_neighbours())
      return;

    wait(_comm0, request);
  }

  /// @brief Start a non-blocking neighbourhood collective exchange of
  /// ghost data with the owning ranks.
  ///
  /// The communication is completed by calling
  /// Scatterer::scatter_rev_end. See ::remote_indices_block for how to
  /// pack `send_buffer` and ::local_indices_block for how to unpack
  /// `recv_buffer`.
  ///
  /// This is a differently named function rather than an overload of
  /// ::scatter_rev_begin because the underlying type of `MPI_Datatype`
  /// is implementation-defined, and is an integer type in some MPI
  /// implementations, so overloading on `int` and `MPI_Datatype` is not
  /// portably unambiguous.
  ///
  /// @note Collective MPI operation. Every rank in the communicator
  /// must call this function, including ranks without neighbours.
  ///
  /// @note The send and receive buffers must **not** be changed or
  /// accessed until after a call to Scatterer::scatter_rev_end.
  ///
  /// @note The pointers `send_buffer` and `recv_buffer` must be
  /// pointers to the data on the *target device*. E.g., if the send and
  /// receive buffers are allocated on a GPU, the `send_buffer` and
  /// `recv_buffer` should be device pointers.
  ///
  /// @param[in] send_buffer Data associated with each ghost index. This
  /// data is sent to the process that owns the index. See
  /// Scatterer::remote_indices_block for the order of the buffer and
  /// how to pack.
  /// @param[in,out] recv_buffer Buffer for storing received data. See
  /// Scatterer::local_indices_block for the order of the buffer and how
  /// to unpack.
  /// @param[in] type MPI datatype for the data associated with one
  /// index, e.g. `dolfinx::MPI::Datatype<T>(bs).type()`. Buffer counts
  /// and displacements are in units of `type`, and the same type must
  /// be used on all ranks. MPI keeps a datatype alive until
  /// communication using it has completed, so `type` may be freed as
  /// soon as this function returns.
  /// @param[out] request Handle for tracking the status of the
  /// non-blocking communication. Any value passed in is overwritten,
  /// and `MPI_REQUEST_NULL` is returned when this rank has nothing to
  /// communicate. The same handle must be passed to
  /// Scatterer::scatter_rev_end to complete the communication.
  template <typename T>
  void scatter_rev_begin_dtype(const T* send_buffer, T* recv_buffer,
                               MPI_Datatype type, MPI_Request& request) const
  {
    if (!has_neighbours())
    {
      request = MPI_REQUEST_NULL;
      return;
    }

    int ierr = MPI_Ineighbor_alltoallv(
        send_buffer, _sizes_remote.data(), _displs_remote.data(), type,
        recv_buffer, _sizes_local.data(), _displs_local.data(), type,
        _comm1.comm(), &request);
    dolfinx::MPI::check_error(_comm1.comm(), ierr);
  }

  /// @brief Start a non-blocking neighbourhood collective exchange of
  /// ghost data with the owning ranks.
  ///
  /// As ::scatter_rev_begin_dtype, but with the MPI datatype built from
  /// a block size.
  ///
  /// @param[in] send_buffer Data associated with each ghost index. This
  /// data is sent to the process that owns the index. See
  /// Scatterer::remote_indices_block for the order of the buffer and
  /// how to pack.
  /// @param[in,out] recv_buffer Buffer for storing received data. See
  /// Scatterer::local_indices_block for the order of the buffer and how
  /// to unpack.
  /// @param[in] bs Number of values per index map index (the block
  /// size). The buffers hold `bs` values for each index in
  /// ::remote_indices_block and ::local_indices_block respectively.
  /// @param[out] request Handle for tracking the status of the
  /// non-blocking communication. Any value passed in is overwritten,
  /// and `MPI_REQUEST_NULL` is returned when this rank has nothing to
  /// communicate. The same handle must be passed to
  /// Scatterer::scatter_rev_end to complete the communication.
  template <typename T>
  void scatter_rev_begin(const T* send_buffer, T* recv_buffer, int bs,
                         MPI_Request& request) const
  {
    // Checked here too, to avoid building a datatype that will not be
    // used
    if (!has_neighbours())
    {
      request = MPI_REQUEST_NULL;
      return;
    }

    dolfinx::MPI::Datatype<T> type(bs);
    scatter_rev_begin_dtype(send_buffer, recv_buffer, type.type(), request);
  }

  /// @brief Complete a non-blocking MPI neighbourhood collective send.
  ///
  /// This function completes the communication started by
  /// ::scatter_rev_begin or ::scatter_rev_begin_dtype.
  ///
  /// @note Local completion of the caller's own request, not itself
  /// collective. Every rank that called ::scatter_rev_begin must
  /// still call this before reusing the buffers.
  ///
  /// @param[in,out] request Handle returned by the matching begin
  /// call. Set to `MPI_REQUEST_NULL` once the communication has
  /// completed.
  void scatter_rev_end(MPI_Request& request) const
  {
    if (!has_neighbours())
      return;

    wait(_comm1, request);
  }

  /// @brief Array of indices for packing/unpacking owned data to/from a
  /// send/receive buffer.
  ///
  /// For a forward scatter, the indices are used to copy required
  /// entries in the owned part of the data array into the appropriate
  /// position in a send buffer. For a reverse scatter, indices are used
  /// for assigning (accumulating) the receive buffer values into
  /// the correct position in the owned part of the data array.
  ///
  /// The indices are in blocks, so for a block size `bs` a buffer holds
  /// `bs` values per index and must be `bs * local_indices_block().size()`
  /// long.
  ///
  /// For a forward scatter, if `x` is the owned part of an array and
  /// `send_buffer` is the send buffer, `send_buffer` is packed such
  /// that:
  ///
  ///     auto& idx = scatterer.local_indices_block()
  ///     std::vector<T> send_buffer(bs * idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         for (int j = 0; j < bs; ++j)
  ///             send_buffer[i * bs + j] = x[idx[i] * bs + j];
  ///
  /// For a reverse scatter, if `recv_buffer` is the received buffer,
  /// then `x` is updated by
  ///
  ///     auto& idx = scatterer.local_indices_block()
  ///     std::vector<T> recv_buffer(bs * idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         for (int j = 0; j < bs; ++j)
  ///             x[idx[i] * bs + j]
  ///                 = op(recv_buffer[i * bs + j], x[idx[i] * bs + j]);
  ///
  /// where `op` is a binary operation, e.g. `x[...] = buffer[...]` or
  /// `x[...] += buffer[...]`.
  ///
  /// @return Indices container.
  const container_type& local_indices_block() const noexcept
  {
    return _local_inds;
  }

  /// @brief Array of indices for packing/unpacking ghost data to/from a
  /// send/receive buffer.
  ///
  /// For a forward scatter, the indices are used to unpack received data
  /// into ghost entries. For a reverse scatter, indices are used to pack
  /// ghost entries into the send buffer.
  ///
  /// The indices are in blocks, so for a block size `bs` a buffer holds
  /// `bs` values per index and must be
  /// `bs * remote_indices_block().size()` long.
  ///
  /// For a forward scatter, if `xg` is the ghost part of the data array
  /// and `recv_buffer` is the receive buffer, `xg` is updated as
  ///
  ///     auto& idx = scatterer.remote_indices_block()
  ///     std::vector<T> recv_buffer(bs * idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         for (int j = 0; j < bs; ++j)
  ///             xg[idx[i] * bs + j] = recv_buffer[i * bs + j];
  ///
  /// For a reverse scatter, if `send_buffer` is the send buffer, then
  /// `send_buffer` is packed such that:
  ///
  ///     auto& idx = scatterer.remote_indices_block()
  ///     std::vector<T> send_buffer(bs * idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         for (int j = 0; j < bs; ++j)
  ///             send_buffer[i * bs + j] = xg[idx[i] * bs + j];
  ///
  /// @return Block indices container.
  const container_type& remote_indices_block() const noexcept
  {
    return _remote_inds;
  }

private:
  // False only on a single rank, where _comm0/_comm1 stay MPI_COMM_NULL
  bool has_neighbours() const noexcept
  {
    return _comm0.comm() != MPI_COMM_NULL;
  }

  // Complete a non-blocking request, checking errors against `comm`
  static void wait(const dolfinx::MPI::Comm& comm, MPI_Request& request)
  {
    int ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);
    dolfinx::MPI::check_error(comm.comm(), ierr);
  }

  // Communicator where the source ranks own the indices in the callers
  // halo, and the destination ranks 'ghost' indices owned by the
  // caller. I.e.,
  // - in-edges (src) are from ranks that own my ghosts
  // - out-edges (dest) go to ranks that 'ghost' my owned indices
  dolfinx::MPI::Comm _comm0{MPI_COMM_NULL};

  // Communicator where the source ranks have ghost indices that are
  // owned by the caller, and the destination ranks are the owners of
  // indices in the callers halo region. I.e.,
  // - in-edges (src) are from ranks that 'ghost' my owned indices
  // - out-edges (dest) are to the owning ranks of my ghost indices
  dolfinx::MPI::Comm _comm1{MPI_COMM_NULL};

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
