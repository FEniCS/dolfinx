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
#include <concepts>
#include <functional>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace dolfinx::common
{
/// @brief Types of MPI communication pattern that can be used by a
/// Scatterer.
enum class ScattererType : std::uint8_t
{
  neighbor, // MPI neighborhood collectives
  p2p       // MPI Isend/Irecv for communication
};

/// @brief A Scatterer supports the MPI scattering and gathering of data
/// that is associated with a common::IndexMap.
///
/// Scatter and gather operations use (i) MPI neighbourhood collective
/// or (ii) non-blocking point-to-point communication modes. The mode is
/// selectable. implementation is designed for sparse communication
/// patterns, as it typical of patterns based on an IndexMap.
///
/// @tparam Container Container type for storing 'local' and 'remote'
/// indices. On CPUs this is normally `std::vector<std::int32_t>`. For
/// GPUs the container should store the indices on the device.
template <class Container = std::vector<std::int32_t>>
  requires std::is_integral_v<typename Container::value_type>
class Scatterer
{
public:
  /// Container type used to store local and remote indices.
  using container_type = Container;

  /// @brief Create a scatterer for data with a layout described by an
  /// IndexMap, and with a block size.
  ///
  /// @param[in] map Index map that describes the parallel layout of
  /// data.
  /// @param[in] bs Number of values associated with each `map` index
  /// (block size).
  Scatterer(const IndexMap& map, int bs)
      : _src(map.src().begin(), map.src().end()),
        _dest(map.dest().begin(), map.dest().end()),
        _sizes_remote(_src.size(), 0), _displs_remote(_src.size() + 1),
        _sizes_local(_dest.size()), _displs_local(_dest.size() + 1)

  {
    if (dolfinx::MPI::size(map.comm()) == 1)
      return;

    // Check that src and dest ranks are unique and sorted
    assert(std::ranges::is_sorted(_src));
    assert(std::ranges::is_sorted(_dest));

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
    assert(_sizes_remote.size() == _src.size());
    assert(_displs_remote.size() == _src.size() + 1);
    auto begin = owners_sorted.begin();
    for (std::size_t i = 0; i < _src.size(); i++)
    {
      auto upper = std::upper_bound(begin, owners_sorted.end(), _src[i]);
      std::size_t num_ind = std::distance(begin, upper);
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
    assert(_sizes_local.size() == _dest.size());
    assert(_displs_local.size() == _dest.size() + 1);
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
    std::ranges::for_each(recv_buffer, [range](auto idx)
                          { assert(idx >= range[0] and idx < range[1]); });
#endif

    {
      // Scale sizes and displacements by block size
      // auto rescale = [](auto& x, int bs)
      // {
      //   std::ranges::transform(x, x.begin(), [bs](auto e) { return e *= bs;
      //   });
      // };
      // rescale(_sizes_local, bs);
      // rescale(_displs_local, bs);
      // rescale(_sizes_remote, bs);
      // rescale(_displs_remote, bs);
      for (auto& x : {std::ref(_sizes_local), std::ref(_displs_local),
                      std::ref(_sizes_remote), std::ref(_displs_remote)})
      {
        std::ranges::transform(x.get(), x.get().begin(),
                               [bs](auto e) { return e *= bs; });
      }
    }

    {
      // Expand local indices using block size and convert it from
      // global to local numbering
      std::vector<typename container_type::value_type> idx(recv_buffer.size()
                                                           * bs);
      std::int64_t offset = range[0] * bs;
      for (std::size_t i = 0; i < recv_buffer.size(); i++)
        for (int j = 0; j < bs; j++)
          idx[i * bs + j] = (recv_buffer[i] * bs + j) - offset;
      _local_inds = std::move(idx);
    }

    {
      // Expand remote indices using block size
      std::vector<typename container_type::value_type> idx(perm.size() * bs);
      for (std::size_t i = 0; i < perm.size(); i++)
        for (int j = 0; j < bs; j++)
          idx[i * bs + j] = perm[i] * bs + j;
      _remote_inds = std::move(idx);
    }
  }

  /// @brief Start a non-blocking send of owned data to ranks that ghost
  /// the data.
  ///
  /// The communication is completed by calling
  /// Scatterer::scatter_fwd_end. The send and receive buffer should not
  /// be changed until after Scatterer::scatter_fwd_end has been called.
  ///
  /// See ::local_indices for instructions on packing `send_buffer` and
  /// ::remote_indices  for instructions on packing `recv_buffer`.
  ///
  /// @note The pointers `send_buffer` and `recv_buffer` should be
  /// pointers to the data on the target device. E.g., if the send and
  /// receive buffers are allocated on a GPU, the `send_buffer` and
  /// `recv_buffer` should be device pointers.
  ///
  /// @param[in] send_buffer Packed local data associated with each
  /// owned local index to be sent to process where the data is ghosted.
  /// It must not be changed until after a call to
  /// Scatterer::scatter_end. The order of the data in the buffer is
  /// given by Scatterer::local_indices.
  /// @param[in,out] recv_buffer Buffer for storing received data. The
  /// position of ghost entries in the buffer is given by
  /// Scatterer::remote_indices. The buffer must not be changed until
  /// after the matching call to Scatterer::scatter_end.
  /// @param[in] requests MPI request handle for tracking the status of
  /// the non-blocking communication. For ScattererType::neighbor
  /// communication it should have size 1. For ScattererType::p2p
  /// communication it should have size IndexMap::dest::size +
  /// IndexMap::src::size.
  /// @param[in] type Type of MPI communication pattern to use.
  template <typename T>
  void scatter_fwd_begin(const T* send_buffer, T* recv_buffer,
                         std::span<MPI_Request> requests,
                         ScattererType type = ScattererType::neighbor) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    if (type == ScattererType::neighbor and requests.size() != 1)
    {
      throw std::runtime_error("Neighborhood collective scatterer should have "
                               "only one MPI_Request.");
    }
    if (type == ScattererType::p2p
        and requests.size() != _dest.size() + _src.size())
    {
      throw std::runtime_error(
          "Point-to-point scatterer has wrong number of MPI_Requests.");
    }

    switch (type)
    {
    case ScattererType::neighbor:
    {
      MPI_Ineighbor_alltoallv(send_buffer, _sizes_local.data(),
                              _displs_local.data(), dolfinx::MPI::mpi_t<T>,
                              recv_buffer, _sizes_remote.data(),
                              _displs_remote.data(), dolfinx::MPI::mpi_t<T>,
                              _comm0.comm(), requests.data());
      break;
    }
    case ScattererType::p2p:
    {
      for (std::size_t i = 0; i < _src.size(); ++i)
      {
        MPI_Irecv(recv_buffer + _displs_remote[i], _sizes_remote[i],
                  dolfinx::MPI::mpi_t<T>, _src[i], MPI_ANY_TAG, _comm0.comm(),
                  &requests[i]);
      }

      for (std::size_t i = 0; i < _dest.size(); ++i)
      {
        MPI_Isend(send_buffer + _displs_local[i], _sizes_local[i],
                  dolfinx::MPI::mpi_t<T>, _dest[i], 0, _comm0.comm(),
                  &requests[i + _src.size()]);
      }
      break;
    }
    default:
      throw std::runtime_error("ScattererType not recognized");
    }
  }

  /// @brief Start a non-blocking send of ghost data to ranks that own
  /// the data.
  ///
  /// The communication is completed by calling Scatterer::scatter_end.
  /// The send and receive buffers must not be changed until after
  /// Scatterer::scatter_end has been called.
  ///
  /// See ::remote_indices for instructions on packing `send_buffer` and
  /// ::local_indices for instructions on packing `recv_buffer`.
  ///
  /// @note The pointers `send_buffer` and `recv_buffer` should be
  /// pointers to the data on the target device. E.g., if the send and
  /// receive buffers are allocated on a GPU, the `send_buffer` and
  /// `recv_buffer` should be device pointers.
  ///
  /// @param[in] send_buffer Data associated with each ghost index. This
  /// data is sent to process that owns the index. It must not be
  /// changed until after a call to Scatterer::scatter_end.
  /// @param[in,out] recv_buffer Buffer used for the received data. The
  /// position of owned indices in the buffer is given by
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
  ///
  /// @param[in] requests MPI request handle for tracking the status of the
  /// non-blocking communication.
  /// @param[in] type Type of MPI communication pattern used by the
  /// Scatterer, either ScattererType::neighbor or ScattererType::p2p.
  template <typename T>
  void scatter_rev_begin(const T* send_buffer, T* recv_buffer,
                         std::span<MPI_Request> requests,
                         ScattererType type = ScattererType::neighbor) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    if (type == ScattererType::neighbor and requests.size() != 1)
    {
      throw std::runtime_error("Neighborhood collective scatterer should have "
                               "only one MPI_Request.");
    }
    if (type == ScattererType::p2p
        and requests.size() != _dest.size() + _src.size())
    {
      throw std::runtime_error(
          "Point-to-point scatterer has wrong number of MPI_Requests.");
    }

    switch (type)
    {
    case ScattererType::neighbor:
    {
      MPI_Ineighbor_alltoallv(send_buffer, _sizes_remote.data(),
                              _displs_remote.data(), MPI::mpi_t<T>, recv_buffer,
                              _sizes_local.data(), _displs_local.data(),
                              MPI::mpi_t<T>, _comm1.comm(), &requests[0]);
      break;
    }
    case ScattererType::p2p:
    {

      // Start non-blocking send from this process to ghost owners
      for (std::size_t i = 0; i < _dest.size(); i++)
      {
        MPI_Irecv(recv_buffer + _displs_local[i], _sizes_local[i],
                  dolfinx::MPI::mpi_t<T>, _dest[i], MPI_ANY_TAG, _comm0.comm(),
                  &requests[i]);
      }

      // Start non-blocking receive from neighbor process for which an
      // owned index is a ghost
      for (std::size_t i = 0; i < _src.size(); i++)
      {
        MPI_Isend(send_buffer + _displs_remote[i], _sizes_remote[i],
                  dolfinx::MPI::mpi_t<T>, _src[i], 0, _comm0.comm(),
                  &requests[i + _dest.size()]);
      }
      break;
    }
    default:
      throw std::runtime_error("ScattererType not recognized");
    }
  }

  /// @brief Complete a non-blocking send from the local owner to
  /// process ranks that have the index as a ghost.
  ///
  /// This function completes the communication started by
  /// ::scatter_fwd_begin or ::scatter_fwd_end.
  ///
  /// @param[in] requests MPI request handle for tracking the status of
  /// the send.
  void scatter_end(std::span<MPI_Request> requests) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_sizes_local.empty() and _sizes_remote.empty())
      return;

    // Wait for communication to complete
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  }

  // /// @brief Size of buffer for packed local data (owned data that is
  // /// shared) used in forward and reverse scatters.
  // ///
  // /// @return Required buffer size.
  // std::size_t local_buffer_size() const noexcept { return _local_inds.size();
  // }

  // /// @brief Buffer size for packed ghost (data shared by owned by
  // /// another process) data used in forward and reverse scatters.
  // ///
  // /// @return Required buffer size.
  // std::size_t remote_buffer_size() const noexcept
  // {
  //   return _remote_inds.size();
  // }

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
  /// @return Indices container
  const container_type& local_indices() const noexcept { return _local_inds; }

  /// @brief Array of indices for packing/unpacking ghost data to/from a
  /// send/receive buffer.
  ///
  /// For a forward scatter, the indices are to copy required entries in
  /// the owned array into the appropriate position in a send buffer.
  /// For a reverse scatter, indices are used for assigning
  /// (accumulating) the receive buffer values into correct position in
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
  ///     auto& idx = scatterer.local_indices()
  ///     std::vector<T> send_buffer(idx.size())
  ///     for (std::size_t i = 0; i < idx.size(); ++i)
  ///         send_buffer[i] = xg[idx[i]];
  ///
  /// The indices are grouped by neighbor process (ghost owners) blocks.
  const container_type& remote_indices() const noexcept { return _remote_inds; }

  /// @brief Create a vector of MPI_Requests for a given
  /// ::ScattererType.
  ///
  /// @return Vector of MPI requests.
  std::vector<MPI_Request> create_requests(ScattererType type
                                           = ScattererType::neighbor)
  {
    switch (type)
    {
    case ScattererType::neighbor:
      return {MPI_REQUEST_NULL};
      break;
    case ScattererType::p2p:
      return std::vector<MPI_Request>(_dest.size() + _src.size(),
                                      MPI_REQUEST_NULL);
    default:
      throw std::runtime_error("ScattererType not recognized");
    }
  }

private:
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

  // Set of ranks that own ghosts
  // FIXME: Should we store the index map instead?
  std::vector<int> _src;

  // Set of ranks ghost owned indices
  // FIXME: Should we store the index map instead?
  std::vector<int> _dest;

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
