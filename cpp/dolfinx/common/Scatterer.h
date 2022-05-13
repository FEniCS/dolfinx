// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/IndexMapNew.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <mpi.h>

using namespace dolfinx;

namespace
{
template <typename T>
void debug_vector(const std::vector<T>& vec)
{
  for (int i = 0; i < dolfinx::MPI::size(MPI_COMM_WORLD); i++)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    if (int rank = dolfinx::MPI::rank(MPI_COMM_WORLD); rank == i)
    {
      std::cout << "\n Rank " << i << std::endl;
      for (auto e : vec)
        std::cout << e << " ";
      std::cout << std::endl;
    }
  }
}
} // namespace

namespace dolfinx::common
{

/// Manage MPI data scatter and gather
class Scatterer
{
public:
  /// Constructor
  Scatterer(const std::shared_ptr<const common::IndexMap> map, int bs)
      : _bs(bs), _comm_owner_to_ghost(MPI_COMM_NULL),
        _comm_ghost_to_owner(MPI_COMM_NULL)
  {
    if (map->overlapped())
    {
      // Ger source (owner of ghosts) and destination (processes that ghosts an
      // owned index) ranks
      const std::vector<int>& src_ranks = map->src();
      const std::vector<int>& dest_ranks = map->dest();

      // We assume src and dest ranks are unique and sorted
      assert(std::is_sorted(src_ranks.begin(), src_ranks.end()));
      assert(std::is_sorted(src_ranks.begin(), src_ranks.end()));

      // Create communicators with directed edges:
      // (0) owner -> ghost,
      // (1) ghost -> owner

      // NOTE: create uniform weights as a workaround to issue
      // https://github.com/pmodels/mpich/issues/5764
      std::vector<int> src_weights(src_ranks.size(), 1);
      std::vector<int> dest_weights(dest_ranks.size(), 1);

      MPI_Comm comm0;
      MPI_Dist_graph_create_adjacent(
          map->comm(), src_ranks.size(), src_ranks.data(), src_weights.data(),
          dest_ranks.size(), dest_ranks.data(), dest_weights.data(),
          MPI_INFO_NULL, false, &comm0);
      _comm_owner_to_ghost = dolfinx::MPI::Comm(comm0, false);

      MPI_Comm comm1;
      MPI_Dist_graph_create_adjacent(
          map->comm(), dest_ranks.size(), dest_ranks.data(),
          dest_weights.data(), src_ranks.size(), src_ranks.data(),
          src_weights.data(), MPI_INFO_NULL, false, &comm1);
      _comm_ghost_to_owner = dolfinx::MPI::Comm(comm1, false);

      // Compute shared indices and group by neighboring (processes for which
      // an index is a ghost)
      const std::vector<int>& owners = map->owners();
      const std::vector<std::int64_t>& ghosts = map->ghosts();
      std::vector<std::int32_t> perm(owners.size());
      std::iota(perm.begin(), perm.end(), 0);
      dolfinx::argsort_radix<std::int32_t>(owners, perm);

      std::vector<int> owners_sorted(owners.size());
      std::vector<std::int64_t> ghosts_sorted(owners.size());

      std::transform(perm.begin(), perm.end(), owners_sorted.begin(),
                     [&owners](auto idx) { return owners[idx]; });
      std::transform(perm.begin(), perm.end(), ghosts_sorted.begin(),
                     [&ghosts](auto idx) { return ghosts[idx]; });

      _sizes_remote.resize(src_ranks.size(), 0);
      _displs_remote.resize(src_ranks.size() + 1, 0);

      std::vector<std::int32_t>::iterator begin = owners_sorted.begin();
      for (std::size_t i = 0; i < src_ranks.size(); i++)
      {
        auto upper = std::upper_bound(begin, owners_sorted.end(), src_ranks[i]);
        int num_ind = std::distance(begin, upper);
        _displs_remote[i + 1] = _displs_remote[i] + num_ind;
        _sizes_remote[i] = num_ind;
        begin = upper;
      }

      _sizes_local.resize(dest_ranks.size());
      _displs_local.resize(_sizes_local.size() + 1);

      MPI_Neighbor_alltoall(_sizes_remote.data(), 1,
                            MPI::mpi_type<std::int32_t>(), _sizes_local.data(),
                            1, MPI::mpi_type<std::int32_t>(),
                            _comm_ghost_to_owner.comm());

      std::partial_sum(_sizes_local.begin(), _sizes_local.end(),
                       std::next(_displs_local.begin()));

      // Send ghost indices to owner, and receive owned indices
      std::vector<std::int64_t> recv_buffer(_displs_local.back(), 0);
      assert((std::int32_t)ghosts_sorted.size() == _displs_remote.back());
      assert((std::int32_t)ghosts_sorted.size() == _displs_remote.back());

      MPI_Neighbor_alltoallv(
          ghosts_sorted.data(), _sizes_remote.data(), _displs_remote.data(),
          MPI::mpi_type<std::int64_t>(), recv_buffer.data(),
          _sizes_local.data(), _displs_local.data(),
          MPI::mpi_type<std::int64_t>(), _comm_ghost_to_owner.comm());

      std::array<std::int64_t, 2> range = map->local_range();

#ifndef NDEBUG
      std::for_each(recv_buffer.begin(), recv_buffer.end(),
                    [&range](auto idx)
                    { assert(idx >= range[0] and idx < range[1]); });
#endif

      // Scale sizes and displacements by block size
      auto scale = [bs = _bs](auto& e) { e *= bs; };
      std::for_each(_sizes_local.begin(), _sizes_local.end(), scale);
      std::for_each(_displs_local.begin(), _displs_local.end(), scale);
      std::for_each(_sizes_remote.begin(), _sizes_remote.end(), scale);
      std::for_each(_displs_remote.begin(), _displs_remote.end(), scale);

      // Expand local indices using block size and convert it from global to
      // local numbering
      _local_inds.resize(_displs_local.back() * _bs);
      std::int64_t offset = range[0] * _bs;
      for (std::size_t i = 0; i < recv_buffer.size(); i++)
        for (int j = 0; j < _bs; j++)
          _local_inds[i * _bs + j] = (recv_buffer[i] * _bs + j) - offset;

      // Expand remote indices using block size
      _remote_inds.resize(perm.size() * _bs);
      for (std::size_t i = 0; i < perm.size(); i++)
        for (int j = 0; j < _bs; j++)
          _remote_inds[i * _bs + j] = perm[i] * _bs + j;
    }
  }

  /// TODO
  template <typename T>
  void scatter_fwd_begin(const xtl::span<const T>& send_buffer,
                         const xtl::span<T>& recv_buffer,
                         MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_displs_local.size() == 1 and _displs_remote.size() == 1)
      return;

    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_local.data(),
                            _displs_local.data(), MPI::mpi_type<T>(),
                            recv_buffer.data(), _sizes_remote.data(),
                            _displs_remote.data(), MPI::mpi_type<T>(),
                            _comm_owner_to_ghost.comm(), &request);
  }

  /// Complete a non-blocking send from the local owner of to process
  /// ranks that have the index as a ghost. This function complete the
  /// communication started by VectorScatter::scatter_fwd_begin.
  ///
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  void scatter_fwd_end(MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_displs_local.size() == 1 and _displs_remote.size() == 1)
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  /// TODO: Add documentation
  // NOTE: This function is not MPI-X friendly
  template <typename T, typename Functor1, typename Functor2>
  void scatter_fwd(const xtl::span<const T>& local_data,
                   xtl::span<T> remote_data, Functor1 pack_fn,
                   Functor2 unpack_fn) const
  {
    std::vector<T> send_buffer(_local_inds.size());
    pack_fn(local_data, _local_inds, send_buffer);

    MPI_Request request;
    std::vector<T> buffer_recv(_displs_remote.back());
    scatter_fwd_begin(xtl::span<const T>(send_buffer),
                      xtl::span<T>(buffer_recv), &request);
    scatter_fwd_end(request);

    // Insert op
    auto op = [](T /*a*/, T b) { return b; };
    unpack_fn(buffer_recv, _remote_inds, remote_data, op);
  }

  /// Start a non-blocking send of ghost values to the owning rank.
  template <typename T>
  void scatter_rev_begin(const xtl::span<const T>& send_buffer,
                         const xtl::span<T>& recv_buffer,
                         MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_displs_local.size() == 1 and _displs_remote.size() == 1)
      return;

    // Send and receive data
    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_remote.data(),
                            _displs_remote.data(), MPI::mpi_type<T>(),
                            recv_buffer.data(), _sizes_local.data(),
                            _displs_local.data(), MPI::mpi_type<T>(),
                            _comm_ghost_to_owner.comm(), &request);
  }

  /// TODO
  void scatter_rev_end(MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    if (_displs_local.size() == 1 and _displs_remote.size() == 1)
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  /// Local pack function
  // TODO: add documentation
  static auto pack()
  {
    return [](const auto& in, const auto& idx, auto& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
  }

  /// Local unpack function
  // TODO: add documentation
  static auto unpack()
  {
    return [](const auto& in, const auto& idx, auto& out, auto op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };
  }

  /// Send n values for each ghost index to owning to the process
  ///
  /*
  /// @param[in,out] local_data Local data associated with each owned
  /// local index to be sent to process where the data is ghosted. Size
  /// must be n * size_local().
  /// @param[in] remote_data Ghost data on this process received from
  /// the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  /// @param[in] op Sum or set received values in local_data
  */
  template <typename T, typename BinaryOp, typename Functor1, typename Functor2>
  void scatter_rev(
      xtl::span<T> local_data, const xtl::span<const T>& remote_data,
      BinaryOp op,
      Functor1 pack_fn =
          [](const auto& in, const auto& idx, auto& out)
      {
        for (std::size_t i = 0; i < idx.size(); ++i)
          out[i] = in[idx[i]];
      },
      Functor2 unpack_fn =
          [](const auto& in, const auto& idx, auto& out, auto op)
      {
        for (std::size_t i = 0; i < idx.size(); ++i)
          out[idx[i]] = op(out[idx[i]], in[i]);
      }) const
  //  Functor1 pack_fn = pack(),
  //  Functor2 unpack_fn = unpack()) const
  {
    // Pack send buffer
    std::vector<T> buffer_send(_displs_remote.back());
    pack_fn(remote_data, _remote_inds, buffer_send);

    // Exchange data
    MPI_Request request;
    std::vector<T> buffer_recv(_local_inds.size());
    scatter_rev_begin(xtl::span<const T>(buffer_send),
                      xtl::span<T>(buffer_recv), request);
    scatter_rev_end(request);

    // Copy or accumulate into "local_data"
    unpack_fn(buffer_recv, _local_inds, local_data, op);
  }

  /// TODO
  const std::vector<std::int32_t>& local_shared_indices() const noexcept
  {
    return _local_inds;
  }

  /// TODO
  const std::vector<std::int32_t>& remote_indices() const noexcept
  {
    return _remote_inds;
  }

  /// TODO
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

  // TODO: add documentation
  std::vector<std::int32_t> _remote_inds;

  // TODO: add documentation
  std::vector<std::int32_t> _sizes_remote;

  // TODO: add documentation
  std::vector<std::int32_t> _displs_remote;

  // TODO: add documentation
  std::vector<std::int32_t> _local_inds;

  // TODO: add documentation
  std::vector<std::int32_t> _sizes_local;

  // TODO: add documentation
  std::vector<std::int32_t> _displs_local;
};
} // namespace dolfinx::common