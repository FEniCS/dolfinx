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
void [[maybe_unused]] debug_vector(const std::vector<T>& vec)
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
class Scatterer
{
public:
  Scatterer(const std::shared_ptr<const common::IndexMapNew> map, int bs)
      : _bs(bs), _comm_owner_to_ghost(MPI_COMM_NULL),
        _comm_ghost_to_owner(MPI_COMM_NULL)
  {
    if (map->overlapped())
    {
      // Create communicators with directed edges:
      // (0) owner -> ghost,
      // (1) ghost -> owner

      const std::vector<int>& src_ranks = map->src();
      const std::vector<int>& dest_ranks = map->dest();

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
      std::swap(perm, _remote_inds);

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
      assert(ghosts_sorted.size() == _displs_remote.back());
      assert(ghosts_sorted.size() == _displs_remote.back());

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

      _local_inds.resize(_displs_local.back());
      std::transform(recv_buffer.begin(), recv_buffer.end(),
                     _local_inds.begin(),
                     [&range](auto idx) { return idx - range[0]; });
    }
  }

  /// Start a non-blocking send of owned data to ranks that ghost the
  /// data. The communication is completed by calling
  /// VectorScatter::fwd_end. The send and receive buffer should not
  /// be changed until after VectorScatter::fwd_end has been called.
  ///
  /// @param[in] send_buffer Local data associated with each owned local
  /// index to be sent to process where the data is ghosted. It must not
  /// be changed until after a call to VectorScatter::fwd_end. The
  /// order of data in the buffer is given by
  /// VectorScatter::scatter_fwd_indices.
  /// @param recv_buffer A buffer used for the received data. The
  /// position of ghost entries in the buffer is given by
  /// VectorScatter::scatter_fwd_ghost_positions. The buffer must not be
  /// accessed or changed until after a call to
  /// VectorScatter::fwd_end.
  /// @param requests The MPI request handle for tracking the status of
  /// the non-blocking communication
  template <typename T>
  void scatter_fwd_begin(const xtl::span<const T>& send_buffer,
                         const xtl::span<T>& recv_buffer,
                         std::vector<MPI_Request>& requests) const
  {
    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_local.data(),
                            _displs_local.data(), MPI::mpi_type<T>(),
                            recv_buffer.data(), _sizes_remote.data(),
                            _displs_remote.data(), MPI::mpi_type<T>(),
                            _comm_owner_to_ghost.comm(), requests.data());
  }

  /// Complete a non-blocking send from the local owner of to process
  /// ranks that have the index as a ghost. This function complete the
  /// communication started by VectorScatter::scatter_fwd_begin.
  ///
  /// @param[in] requests The MPI request handle for tracking the status
  /// of the send
  void scatter_fwd_end(std::vector<MPI_Request>& requests) const
  {
    // Wait for communication to complete
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  }

  /// TODO: Add documentation
  // NOTE: This function is not MPI-X friendly
  template <typename T, typename Functor>
  void scatter_fwd(const xtl::span<const T>& local_data,
                   xtl::span<T> remote_data, Functor gather_fn = gather()) const
  {
    std::vector<T> send_buffer(_local_inds.size());
    gather_fn(local_data, _local_inds, send_buffer);

    std::vector<MPI_Request> requests(1);
    std::vector<T> buffer_recv(_displs_remote.back());
    scatter_fwd_begin(xtl::span<const T>(send_buffer),
                      xtl::span<T>(buffer_recv), requests);
    scatter_fwd_end(requests);

    gather_fn(buffer_recv, _remote_inds, remote_data);
  }

  /// Local gather function
  // TODO: add documentation
  static auto gather()
  {
    return [](const auto& in, const auto& idx, auto& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
  }

  /// Local scatter function
  // TODO: add documentation
  static auto scatter()
  {
    return [](const auto& in, const auto& idx, auto& out, auto& op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };
  }

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