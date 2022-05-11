// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/IndexMapNew.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <mpi.h>

using namespace dolfinx;

namespace dolfinx::common
{
class Scatterer
{
public:
  Scatterer(const std::shared_ptr<const common::IndexMapNew> map, int bs)
      : _map(map), _bs(bs), _comm_owner_to_ghost(MPI_COMM_NULL),
        _comm_ghost_to_owner(MPI_COMM_NULL)
  {
    // Create communicators with directed edges:
    // (0) owner -> ghost,
    // (1) ghost -> owner
    if (_map->overlapped())
    {
      const std::vector<int>& src_ranks = map->src();
      const std::vector<int>& dest_ranks = map->src();

      // NOTE: create uniform weights as a workaround to issue
      // https://github.com/pmodels/mpich/issues/5764
      std::vector<int> src_weights(src_ranks.size(), 1);
      std::vector<int> dest_weights(dest_ranks.size(), 1);

      MPI_Comm comm = _map->comm();

      MPI_Comm comm0;
      MPI_Dist_graph_create_adjacent(comm, src_ranks.size(), src_ranks.data(),
                                     src_weights.data(), dest_ranks.size(),
                                     dest_ranks.data(), dest_weights.data(),
                                     MPI_INFO_NULL, false, &comm0);
      _comm_owner_to_ghost = dolfinx::MPI::Comm(comm0, false);

      MPI_Comm comm1;
      MPI_Dist_graph_create_adjacent(comm, dest_ranks.size(), dest_ranks.data(),
                                     dest_weights.data(), src_ranks.size(),
                                     src_ranks.data(), src_weights.data(),
                                     MPI_INFO_NULL, false, &comm1);
      _comm_ghost_to_owner = dolfinx::MPI::Comm(comm1, false);

      // Compute shared indices and group by neighboring (processes for which an
      // index is a ghost)
      
    }
  }

  //   /// Start a non-blocking send of owned data to ranks that ghost the
  //   /// data. The communication is completed by calling
  //   /// VectorScatter::fwd_end. The send and receive buffer should not
  //   /// be changed until after VectorScatter::fwd_end has been called.
  //   ///
  //   /// @param[in] send_buffer Local data associated with each owned local
  //   /// index to be sent to process where the data is ghosted. It must not
  //   /// be changed until after a call to VectorScatter::fwd_end. The
  //   /// order of data in the buffer is given by
  //   /// VectorScatter::scatter_fwd_indices.
  //   /// @param request The MPI request handle for tracking the status of
  //   /// the non-blocking communication
  //   /// @param recv_buffer A buffer used for the received data. The
  //   /// position of ghost entries in the buffer is given by
  //   /// VectorScatter::scatter_fwd_ghost_positions. The buffer must not be
  //   /// accessed or changed until after a call to
  //   /// VectorScatter::fwd_end.
  //   template <typename T>
  //   void scatter_fwd_begin(const xtl::span<const T>& send_buffer,
  //                          const xtl::span<T>& recv_buffer,
  //                          std::vector<MPI_Request>& request) const
  //   {
  //     // Send displacement
  //     const std::vector<int32_t>& displs_send_fwd =
  //     _shared_indices->offsets(); assert(send_buffer.size() ==
  //     std::size_t(displs_send_fwd.back()));

  //     MPI_Ineighbor_alltoallv(
  //         send_buffer.data(), _sizes_send_fwd.data(), displs_send_fwd.data(),
  //         MPI::mpi_type<T>(), recv_buffer.data(), _sizes_recv_fwd.data(),
  //         _displs_recv_fwd.data(), MPI::mpi_type<T>(),
  //         _map->comm(IndexMap::Direction::forward), request.data());
  //   }

private:
  // Map describing the data layout
  std::shared_ptr<const common::IndexMapNew> _map;

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
};
} // namespace dolfinx::common