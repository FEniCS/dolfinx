// Copyright (C) 2022 Igor Baratta and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "IndexMap.h"
#include "MPI.h"
#include "sort.h"
#include <algorithm>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx::common
{
/// @brief A Scatterer supports the MPI scattering and gathering of data
/// that is associated with a common::IndexMap.
///
/// Scatter and gather operations uses MPI neighbourhood collectives.
/// The implementation is designed is for sparse communication patterns,
/// as it typical of patterns based on and IndexMap.
template <class Allocator = std::allocator<std::int32_t>>
class Scatterer
{
public:
  /// The allocator type
  using allocator_type = Allocator;

  /// Types of MPI communication pattern used by the Scatterer.
  enum class type
  {
    neighbor, // use MPI neighborhood collectives
    p2p       // use MPI Isend/Irecv for communication
  };

  /// @brief Create a scatterer
  /// @param[in] map The index map that describes the parallel layout of
  /// data.
  /// @param[in] bs The block size of data associated with each index in
  /// `map` that will be scattered/gathered.
  /// @param[in] alloc The memory allocator for indices.
  Scatterer(const IndexMap& map, int bs, const Allocator& alloc = Allocator())
      : _bs(bs), _remote_inds(0, alloc), _local_inds(0, alloc), _src(map.src()),
        _dest(map.dest())
  {
    if (map.overlapped())
    {
      // Check that src and dest ranks are unique and sorted
      assert(std::is_sorted(_src.begin(), _src.end()));
      assert(std::is_sorted(_dest.begin(), _dest.end()));

      // Create communicators with directed edges:
      // (0) owner -> ghost,
      // (1) ghost -> owner
      MPI_Comm comm0;
      MPI_Dist_graph_create_adjacent(
          map.comm(), _src.size(), _src.data(), MPI_UNWEIGHTED, _dest.size(),
          _dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
      _comm0 = dolfinx::MPI::Comm(comm0, false);

      MPI_Comm comm1;
      MPI_Dist_graph_create_adjacent(
          map.comm(), _dest.size(), _dest.data(), MPI_UNWEIGHTED, _src.size(),
          _src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);
      _comm1 = dolfinx::MPI::Comm(comm1, false);

      // Build permutation array that sorts ghost indices by owning rank
      const std::vector<int>& owners = map.owners();
      std::vector<std::int32_t> perm(owners.size());
      std::iota(perm.begin(), perm.end(), 0);
      dolfinx::argsort_radix<std::int32_t>(owners, perm);

      // Sort (i) ghost indices and (ii) ghost index owners by rank
      // (using perm array)
      const std::vector<std::int64_t>& ghosts = map.ghosts();
      std::vector<int> owners_sorted(owners.size());
      std::vector<std::int64_t> ghosts_sorted(owners.size());
      std::transform(perm.begin(), perm.end(), owners_sorted.begin(),
                     [&owners](auto idx) { return owners[idx]; });
      std::transform(perm.begin(), perm.end(), ghosts_sorted.begin(),
                     [&ghosts](auto idx) { return ghosts[idx]; });

      // For data associated with ghost indices, packed by owning
      // (neighbourhood) rank, compute sizes and displacements. I.e.,
      // when sending ghost index data from this rank to the owning
      // ranks, disp[i] is the first entry in the buffer sent to
      // neighbourhood rank i, and disp[i + 1] - disp[i] is the number
      // of values sent to rank i.
      _sizes_remote.resize(_src.size(), 0);
      _displs_remote.resize(_src.size() + 1, 0);
      std::vector<std::int32_t>::iterator begin = owners_sorted.begin();
      for (std::size_t i = 0; i < _src.size(); i++)
      {
        auto upper = std::upper_bound(begin, owners_sorted.end(), _src[i]);
        int num_ind = std::distance(begin, upper);
        _displs_remote[i + 1] = _displs_remote[i] + num_ind;
        _sizes_remote[i] = num_ind;
        begin = upper;
      }

      // For data associated with owned indices that are ghosted by
      // other ranks, compute the size and displacement arrays. When
      // sending data associated with ghost indices to the owner, these
      // size and displacement arrays are for the receive buffer.

      // Compute sizes and displacements of local data (how many local
      // elements to be sent/received grouped by neighbors)
      _sizes_local.resize(_dest.size());
      _displs_local.resize(_sizes_local.size() + 1);
      _sizes_remote.reserve(1);
      _sizes_local.reserve(1);
      MPI_Neighbor_alltoall(_sizes_remote.data(), 1, MPI_INT32_T,
                            _sizes_local.data(), 1, MPI_INT32_T, _comm1.comm());
      std::partial_sum(_sizes_local.begin(), _sizes_local.end(),
                       std::next(_displs_local.begin()));

      assert((std::int32_t)ghosts_sorted.size() == _displs_remote.back());
      assert((std::int32_t)ghosts_sorted.size() == _displs_remote.back());

      // Send ghost global indices to owning rank, and receive owned
      // indices that are ghosts on other ranks
      std::vector<std::int64_t> recv_buffer(_displs_local.back(), 0);
      MPI_Neighbor_alltoallv(ghosts_sorted.data(), _sizes_remote.data(),
                             _displs_remote.data(), MPI_INT64_T,
                             recv_buffer.data(), _sizes_local.data(),
                             _displs_local.data(), MPI_INT64_T, _comm1.comm());

      const std::array<std::int64_t, 2> range = map.local_range();
#ifndef NDEBUG
      // Check that all received indice are within the owned range
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
      _local_inds = std::vector<std::int32_t, allocator_type>(
          recv_buffer.size() * _bs, alloc);
      std::int64_t offset = range[0] * _bs;
      for (std::size_t i = 0; i < recv_buffer.size(); i++)
        for (int j = 0; j < _bs; j++)
          _local_inds[i * _bs + j] = (recv_buffer[i] * _bs + j) - offset;

      // Expand remote indices using block size
      _remote_inds
          = std::vector<std::int32_t, allocator_type>(perm.size() * _bs, alloc);
      for (std::size_t i = 0; i < perm.size(); i++)
        for (int j = 0; j < _bs; j++)
          _remote_inds[i * _bs + j] = perm[i] * _bs + j;
    }
  }

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
  /// @param requests The MPI request handle for tracking the status of
  /// the non-blocking communication
  /// @param[in] type The type of MPI communication pattern used by the
  /// Scatterer, either Scatterer::type::neighbor or Scatterer::type::p2p.
  template <typename T>
  void scatter_fwd_begin(std::span<const T> send_buffer,
                         std::span<T> recv_buffer,
                         std::span<MPI_Request> requests,
                         Scatterer::type type = type::neighbor) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    switch (type)
    {
    case type::neighbor:
    {
      assert(requests.size() == std::size_t(1));
      MPI_Ineighbor_alltoallv(
          send_buffer.data(), _sizes_local.data(), _displs_local.data(),
          dolfinx::MPI::mpi_type<T>(), recv_buffer.data(), _sizes_remote.data(),
          _displs_remote.data(), dolfinx::MPI::mpi_type<T>(), _comm0.comm(),
          requests.data());
      break;
    }
    case type::p2p:
    {
      assert(requests.size() == _dest.size() + _src.size());
      for (std::size_t i = 0; i < _src.size(); i++)
      {
        MPI_Irecv(recv_buffer.data() + _displs_remote[i], _sizes_remote[i],
                  dolfinx::MPI::mpi_type<T>(), _src[i], MPI_ANY_TAG,
                  _comm0.comm(), &requests[i]);
      }

      for (std::size_t i = 0; i < _dest.size(); i++)
      {
        MPI_Isend(send_buffer.data() + _displs_local[i], _sizes_local[i],
                  dolfinx::MPI::mpi_type<T>(), _dest[i], 0, _comm0.comm(),
                  &requests[i + _src.size()]);
      }
      break;
    }
    default:
      throw std::runtime_error("Scatter::type not recognized");
    }
  }

  /// @brief Complete a non-blocking send from the local owner to
  /// process ranks that have the index as a ghost.
  ///
  /// This function completes the communication started by
  /// Scatterer::scatter_fwd_begin.
  ///
  /// @param[in] requests The MPI request handle for tracking the status
  /// of the send
  void scatter_fwd_end(std::span<MPI_Request> requests) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Wait for communication to complete
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  }

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
  /// @param[in] requests The MPI request handle for tracking the status
  /// of the send
  /// @param[in] type The type of MPI communication pattern used by the
  /// Scatterer, either Scatterer::type::neighbor or Scatterer::type::p2p.
  template <typename T, typename Functor>
  void scatter_fwd_begin(std::span<const T> local_data,
                         std::span<T> local_buffer, std::span<T> remote_buffer,
                         Functor pack_fn, std::span<MPI_Request> requests,
                         Scatterer::type type = type::neighbor) const
  {
    assert(local_buffer.size() == _local_inds.size());
    assert(remote_buffer.size() == _remote_inds.size());
    pack_fn(local_data, _local_inds, local_buffer);
    scatter_fwd_begin(std::span<const T>(local_buffer), remote_buffer, requests,
                      type);
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
  /// @param[in] requests The MPI request handle for tracking the status
  /// of the send
  template <typename T, typename Functor>
  void scatter_fwd_end(std::span<const T> remote_buffer,
                       std::span<T> remote_data, Functor unpack_fn,
                       std::span<MPI_Request> requests) const
  {
    assert(remote_buffer.size() == _remote_inds.size());
    assert(remote_data.size() == _remote_inds.size());
    scatter_fwd_end(requests);
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
  void scatter_fwd(std::span<const T> local_data,
                   std::span<T> remote_data) const
  {
    std::vector<MPI_Request> requests(1, MPI_REQUEST_NULL);
    std::vector<T> local_buffer(local_buffer_size(), 0);
    std::vector<T> remote_buffer(remote_buffer_size(), 0);
    auto pack_fn = [](const auto& in, const auto& idx, auto& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
    scatter_fwd_begin(local_data, std::span<T>(local_buffer),
                      std::span<T>(remote_buffer), pack_fn,
                      std::span<MPI_Request>(requests));

    auto unpack_fn = [](const auto& in, const auto& idx, auto& out, auto op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };

    scatter_fwd_end(std::span<const T>(remote_buffer), remote_data, unpack_fn,
                    std::span<MPI_Request>(requests));
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
  /// @param requests The MPI request handle for tracking the status of
  /// the non-blocking communication
  /// @param[in] type The type of MPI communication pattern used by the
  /// Scatterer, either Scatterer::type::neighbor or Scatterer::type::p2p.
  template <typename T>
  void scatter_rev_begin(std::span<const T> send_buffer,
                         std::span<T> recv_buffer,
                         std::span<MPI_Request> requests,
                         Scatterer::type type = type::neighbor) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // // Send and receive data

    switch (type)
    {
    case type::neighbor:
    {
      assert(requests.size() == 1);
      MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_remote.data(),
                              _displs_remote.data(), MPI::mpi_type<T>(),
                              recv_buffer.data(), _sizes_local.data(),
                              _displs_local.data(), MPI::mpi_type<T>(),
                              _comm1.comm(), &requests[0]);
      break;
    }
    case type::p2p:
    {
      assert(requests.size() == _dest.size() + _src.size());
      // Start non-blocking send from this process to ghost owners.
      for (std::size_t i = 0; i < _dest.size(); i++)
      {
        MPI_Irecv(recv_buffer.data() + _displs_local[i], _sizes_local[i],
                  dolfinx::MPI::mpi_type<T>(), _dest[i], MPI_ANY_TAG,
                  _comm0.comm(), &requests[i]);
      }

      // Start non-blocking receive from neighbor process for which an owned
      // index is a ghost.
      for (std::size_t i = 0; i < _src.size(); i++)
      {
        MPI_Isend(send_buffer.data() + _displs_remote[i], _sizes_remote[i],
                  dolfinx::MPI::mpi_type<T>(), _src[i], 0, _comm0.comm(),
                  &requests[i + _dest.size()]);
      }
      break;
    }
    default:
      throw std::runtime_error("Scatter::type not recognized");
    }
  }

  /// @brief End the reverse scatter communication.
  ///
  /// This function must be called after Scatterer::scatter_rev_begin.
  /// The buffers passed to Scatterer::scatter_rev_begin must not be
  /// modified until after the function has been called.
  ///
  /// @param[in] request The handle used when calling
  /// Scatterer::scatter_rev_begin
  void scatter_rev_end(std::span<MPI_Request> request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Wait for communication to complete
    MPI_Waitall(request.size(), request.data(), MPI_STATUS_IGNORE);
  }

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
  /// @param[in] type The type of MPI communication pattern used by the
  /// Scatterer, either Scatterer::type::neighbor or Scatterer::type::p2p.
  template <typename T, typename Functor>
  void scatter_rev_begin(std::span<const T> remote_data,
                         std::span<T> remote_buffer, std::span<T> local_buffer,
                         Functor pack_fn, std::span<MPI_Request> request,
                         Scatterer::type type = type::neighbor) const
  {
    assert(local_buffer.size() == _local_inds.size());
    assert(remote_buffer.size() == _remote_inds.size());
    pack_fn(remote_data, _remote_inds, remote_buffer);
    scatter_rev_begin(std::span<const T>(remote_buffer), local_buffer, request,
                      type);
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
  void scatter_rev_end(std::span<const T> local_buffer, std::span<T> local_data,
                       Functor unpack_fn, BinaryOp op,
                       std::span<MPI_Request> request)
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
  void scatter_rev(std::span<T> local_data, std::span<const T> remote_data,
                   BinaryOp op)
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
    std::vector<MPI_Request> request(1, MPI_REQUEST_NULL);
    scatter_rev_begin(remote_data, std::span<T>(remote_buffer),
                      std::span<T>(local_buffer), pack_fn,
                      std::span<MPI_Request>(request));
    scatter_rev_end(std::span<const T>(local_buffer), local_data, unpack_fn, op,
                    std::span<MPI_Request>(request));
  }

  /// @brief Size of buffer for local data (owned and shared) used in
  /// forward and reverse communication
  /// @return The required buffer size
  std::int32_t local_buffer_size() const noexcept { return _local_inds.size(); }

  /// @brief Buffer size for remote data (ghosts) used in forward and
  /// reverse communication
  /// @return The required buffer size
  std::int32_t remote_buffer_size() const noexcept
  {
    return _remote_inds.size();
  }

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
  /// common::IndexMap use to create the scatterer
  /// @return The block size
  int bs() const noexcept { return _bs; }

  /// @brief Create a vector of MPI_Requests for a given Scatterer::type
  /// @return A vector of MPI requests
  std::vector<MPI_Request> create_request_vector(Scatterer::type type
                                                 = type::neighbor)
  {
    std::vector<MPI_Request> requests;
    switch (type)
    {
    case type::neighbor:
      requests = {MPI_REQUEST_NULL};
      break;
    case type::p2p:
      requests.resize(_dest.size() + _src.size(), MPI_REQUEST_NULL);
      break;
    default:
      throw std::runtime_error("Scatter::type not recognized");
    }
    return requests;
  }

private:
  // Block size
  int _bs;

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
  std::vector<std::int32_t, allocator_type> _remote_inds;

  // Number of remote indices (ghosts) for each neighbor process
  std::vector<int> _sizes_remote;

  // Displacements of remote data for mpi scatter and gather
  std::vector<int> _displs_remote;

  // Permutation indices used to pack and unpack local shared data
  // (owned indices that are shared with other processes). Indices are
  // grouped by neighbor process.
  std::vector<std::int32_t, allocator_type> _local_inds;

  // Number of local shared indices per neighbor process
  std::vector<int> _sizes_local;

  // Displacements of local data for mpi scatter and gather
  std::vector<int> _displs_local;

  // Set of ranks that own ghosts
  // FIXME: Should we store the index map instead?
  std::vector<int> _src;

  // Set of ranks ghost owned indices
  // FIXME: Should we store the index map instead?
  std::vector<int> _dest;
};
} // namespace dolfinx::common
