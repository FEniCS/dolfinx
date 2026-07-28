// Copyright (C) 2007-2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPI.h"
#include <array>
#include <cstddef>
#include <dolfinx/common/log.h>
#include <iostream>
#include <utility>
#include <vector>

//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(MPI_Comm comm, bool duplicate)
{
  // Duplicate communicator
  if (duplicate and comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_dup(comm, &_comm);
    dolfinx::MPI::check_error(comm, err);
  }
  else
    _comm = comm;
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(const Comm& comm) noexcept
    : dolfinx::MPI::Comm::Comm(comm._comm, true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(Comm&& comm) noexcept
{
  this->_comm = comm._comm;
  comm._comm = MPI_COMM_NULL;
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::~Comm()
{
  // Free the comm
  if (_comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_free(&_comm);
    dolfinx::MPI::check_error(_comm, err);
  }
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm&
dolfinx::MPI::Comm::operator=(dolfinx::MPI::Comm&& comm) noexcept
{
  // Free the currently held comm
  if (this->_comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_free(&this->_comm);
    dolfinx::MPI::check_error(this->_comm, err);
  }

  // Move comm from other object
  this->_comm = comm._comm;
  comm._comm = MPI_COMM_NULL;
  return *this;
}
//-----------------------------------------------------------------------------
MPI_Comm dolfinx::MPI::Comm::comm() const noexcept { return _comm; }
//-----------------------------------------------------------------------------
int dolfinx::MPI::rank(const MPI_Comm comm)
{
  int rank;
  int err = MPI_Comm_rank(comm, &rank);
  dolfinx::MPI::check_error(comm, err);
  return rank;
}
//-----------------------------------------------------------------------------
int dolfinx::MPI::size(const MPI_Comm comm)
{
  int size;
  int err = MPI_Comm_size(comm, &size);
  dolfinx::MPI::check_error(comm, err);
  return size;
}
//-----------------------------------------------------------------------------
void dolfinx::MPI::check_error(MPI_Comm comm, int code)
{
  if (code != MPI_SUCCESS)
  {
    int len = MPI_MAX_ERROR_STRING;
    std::string error_string(MPI_MAX_ERROR_STRING, ' ');
    MPI_Error_string(code, error_string.data(), &len);
    error_string.resize(len);
    std::cerr << error_string << '\n';
    MPI_Abort(comm, code);
    std::abort();
  }
}
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_pcx(MPI_Comm comm, std::span<const int> edges)
{
  spdlog::info(
      "Computing communication graph edges (using PCX algorithm). Number "
      "of input edges: {}",
      static_cast<int>(edges.size()));

  // Build array with '0' for no outedge and '1' for an outedge for each
  // rank
  const int size = dolfinx::MPI::size(comm);
  std::vector<int> edge_count_send(size, 0);
  for (auto e : edges)
    edge_count_send[e] = 1;

  // Determine how many in-edges this rank has
  std::vector<int> recvcounts(size, 1);
  int in_edges = 0;
  MPI_Request request_scatter;
  int err = MPI_Ireduce_scatter(edge_count_send.data(), &in_edges,
                                recvcounts.data(), MPI_INT, MPI_SUM, comm,
                                &request_scatter);
  dolfinx::MPI::check_error(comm, err);

  std::vector<MPI_Request> send_requests(edges.size());
  std::vector<std::byte> send_buffer(edges.size());
  for (std::size_t e = 0; e < edges.size(); ++e)
  {
    int err = MPI_Isend(send_buffer.data() + e, 1, MPI_BYTE, edges[e],
                        static_cast<int>(tag::consensus_pcx), comm,
                        &send_requests[e]);
    dolfinx::MPI::check_error(comm, err);
  }

  // Probe for incoming messages and store incoming rank
  err = MPI_Wait(&request_scatter, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(comm, err);
  std::vector<int> other_ranks;
  while (in_edges > 0)
  {
    // Check for message
    int request_pending;
    MPI_Status status;
    int err = MPI_Iprobe(MPI_ANY_SOURCE, static_cast<int>(tag::consensus_pcx),
                         comm, &request_pending, &status);
    dolfinx::MPI::check_error(comm, err);
    if (request_pending)
    {
      // Receive message and store rank
      int other_rank = status.MPI_SOURCE;
      std::byte buffer_recv;
      int err = MPI_Recv(&buffer_recv, 1, MPI_BYTE, other_rank,
                         static_cast<int>(tag::consensus_pcx), comm,
                         MPI_STATUS_IGNORE);
      dolfinx::MPI::check_error(comm, err);
      other_ranks.push_back(other_rank);
      --in_edges;
    }
  }

  // Complete sends before send_buffer is destroyed
  err = MPI_Waitall(send_requests.size(), send_requests.data(),
                    MPI_STATUSES_IGNORE);
  dolfinx::MPI::check_error(comm, err);

  spdlog::info("Finished graph edge discovery using PCX algorithm. Number "
               "of discovered edges {}",
               static_cast<int>(other_ranks.size()));

  return other_ranks;
}
//-----------------------------------------------------------------------------
namespace
{
/// @brief Shared implementation behind both MPI::compute_graph_edges_nbx
/// overloads: run K independent NBX consensus rounds concurrently
/// behind a single shared barrier (K=1 for the plain, single-edge-set
/// case).
///
/// Each edge set is distinguished by its own tag, so K persistent
/// listeners can be posted up front and a message for one round is
/// never mistaken for another.
template <std::size_t K>
std::array<std::vector<int>, K>
nbx_consensus_rounds(MPI_Comm comm, std::array<std::span<const int>, K> edges,
                     std::array<int, K> tags)
{
  // Post a persistent listener per edge set. Posting ahead of arrival
  // (vs. probing reactively) lets MPI treat each as an expected
  // message, avoiding buffering overhead.
  std::array<std::byte, K> buffer_recv;
  std::array<MPI_Request, K> recv_request;
  for (std::size_t i = 0; i < K; ++i)
  {
    int err = MPI_Irecv(&buffer_recv[i], 1, MPI_BYTE, MPI_ANY_SOURCE, tags[i],
                        comm, &recv_request[i]);
    dolfinx::MPI::check_error(comm, err);
  }

  // Synchronised, non-blocking sends; content is never inspected (only
  // arrival matters), so every send shares one source buffer. Pack all
  // K sets into one flat request array so "have all sends completed"
  // below is a single MPI_Testall call.
  std::array<std::size_t, K + 1> send_offset{0};
  for (std::size_t i = 0; i < K; ++i)
    send_offset[i + 1] = send_offset[i] + edges[i].size();
  std::vector<MPI_Request> send_requests(send_offset[K]);
  std::byte send_buffer{0};
  for (std::size_t i = 0; i < K; ++i)
  {
    for (std::size_t e = 0; e < edges[i].size(); ++e)
    {
      int err = MPI_Issend(&send_buffer, 1, MPI_BYTE, edges[i][e], tags[i],
                           comm, &send_requests[send_offset[i] + e]);
      dolfinx::MPI::check_error(comm, err);
    }
  }

  // Ranks that send to this rank, per edge set
  std::array<std::vector<int>, K> src_ranks;

  // Start sending/receiving. A single barrier covers all K sets: it
  // can only start once every send, across all sets, has completed.
  MPI_Request barrier_request;
  bool comm_complete = false;
  bool barrier_active = false;
  while (!comm_complete)
  {
    // Drain all currently queued messages, for each edge set
    for (std::size_t i = 0; i < K; ++i)
    {
      int flag_recv;
      MPI_Status status;
      int err = MPI_Test(&recv_request[i], &flag_recv, &status);
      dolfinx::MPI::check_error(comm, err);
      while (flag_recv)
      {
        src_ranks[i].push_back(status.MPI_SOURCE);
        int err = MPI_Irecv(&buffer_recv[i], 1, MPI_BYTE, MPI_ANY_SOURCE,
                            tags[i], comm, &recv_request[i]);
        dolfinx::MPI::check_error(comm, err);

        err = MPI_Test(&recv_request[i], &flag_recv, &status);
        dolfinx::MPI::check_error(comm, err);
      }
    }

    if (barrier_active)
    {
      // Check for barrier completion
      int flag = 0;
      int err = MPI_Test(&barrier_request, &flag, MPI_STATUS_IGNORE);
      dolfinx::MPI::check_error(comm, err);
      if (flag)
        comm_complete = true;
    }
    else
    {
      // Check if all sends, across all K sets, have completed
      int flag = 0;
      int err = MPI_Testall(send_requests.size(), send_requests.data(), &flag,
                            MPI_STATUSES_IGNORE);
      dolfinx::MPI::check_error(comm, err);
      if (flag)
      {
        // All sends have completed, start non-blocking barrier
        int err = MPI_Ibarrier(comm, &barrier_request);
        dolfinx::MPI::check_error(comm, err);
        barrier_active = true;
      }
    }
  }

  // No more messages can arrive once the barrier has completed, so
  // cancel the still-outstanding listeners. A sender's Issend only
  // requires the receive to be posted, not yet observed by this rank,
  // so cancellation can race with a real match; if so, record it
  // instead of discarding it.
  for (std::size_t i = 0; i < K; ++i)
  {
    int err = MPI_Cancel(&recv_request[i]);
    dolfinx::MPI::check_error(comm, err);
    MPI_Status cancel_status;
    err = MPI_Wait(&recv_request[i], &cancel_status);
    dolfinx::MPI::check_error(comm, err);
    int cancelled;
    MPI_Test_cancelled(&cancel_status, &cancelled);
    if (!cancelled)
      src_ranks[i].push_back(cancel_status.MPI_SOURCE);
  }

  return src_ranks;
}
} // namespace
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_nbx(MPI_Comm comm, std::span<const int> edges,
                                      int tag)
{
  spdlog::info(
      "Computing communication graph edges (using NBX algorithm). Number "
      "of input edges: {}",
      static_cast<int>(edges.size()));

  auto src_ranks = ::nbx_consensus_rounds<1>(comm, {edges}, {tag});

  spdlog::info("Finished graph edge discovery using NBX algorithm. Number "
               "of discovered edges {}",
               static_cast<int>(src_ranks[0].size()));

  return std::move(src_ranks[0]);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
dolfinx::MPI::compute_graph_edges_nbx(MPI_Comm comm,
                                      std::span<const int> edges0, int tag0,
                                      std::span<const int> edges1, int tag1)
{
  assert(tag0 != tag1);
  spdlog::info(
      "Computing communication graph edges for two overlapped consensus "
      "rounds (using NBX algorithm). Number of input edges: {}, {}",
      static_cast<int>(edges0.size()), static_cast<int>(edges1.size()));

  auto src_ranks
      = ::nbx_consensus_rounds<2>(comm, {edges0, edges1}, {tag0, tag1});

  spdlog::info("Finished overlapped graph edge discovery using NBX "
               "algorithm. Number of discovered edges {}, {}",
               static_cast<int>(src_ranks[0].size()),
               static_cast<int>(src_ranks[1].size()));

  return {std::move(src_ranks[0]), std::move(src_ranks[1])};
}
//-----------------------------------------------------------------------------
