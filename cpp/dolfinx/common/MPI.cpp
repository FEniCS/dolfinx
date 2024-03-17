// Copyright (C) 2007-2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPI.h"
#include <dolfinx/common/log.h>
#include <iostream>

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
    std::cerr << error_string << std::endl;
    MPI_Abort(comm, code);
    std::abort();
  }
}
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_pcx(MPI_Comm comm, std::span<const int> edges)
{
  LOG(INFO)
      << "Computing communication graph edges (using PCX algorithm). Number "
         "of input edges: "
      << edges.size();

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

  LOG(INFO) << "Finished graph edge discovery using PCX algorithm. Number "
               "of discovered edges "
            << other_ranks.size();

  return other_ranks;
}
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_nbx(MPI_Comm comm, std::span<const int> edges)
{
  LOG(INFO)
      << "Computing communication graph edges (using NBX algorithm). Number "
         "of input edges: "
      << edges.size();

  // Start non-blocking synchronised send
  std::vector<MPI_Request> send_requests(edges.size());
  std::vector<std::byte> send_buffer(edges.size());
  for (std::size_t e = 0; e < edges.size(); ++e)
  {
    int err = MPI_Issend(send_buffer.data() + e, 1, MPI_BYTE, edges[e],
                         static_cast<int>(tag::consensus_pex), comm,
                         &send_requests[e]);
    dolfinx::MPI::check_error(comm, err);
  }

  // Vector to hold ranks that send data to this rank
  std::vector<int> other_ranks;

  // Start sending/receiving
  MPI_Request barrier_request;
  bool comm_complete = false;
  bool barrier_active = false;
  while (!comm_complete)
  {
    // Check for message
    int request_pending;
    MPI_Status status;
    int err = MPI_Iprobe(MPI_ANY_SOURCE, static_cast<int>(tag::consensus_pex),
                         comm, &request_pending, &status);
    dolfinx::MPI::check_error(comm, err);

    // Check if message is waiting to be processed
    if (request_pending)
    {
      // Receive it
      int other_rank = status.MPI_SOURCE;
      std::byte buffer_recv;
      int err = MPI_Recv(&buffer_recv, 1, MPI_BYTE, other_rank,
                         static_cast<int>(tag::consensus_pex), comm,
                         MPI_STATUS_IGNORE);
      dolfinx::MPI::check_error(comm, err);
      other_ranks.push_back(other_rank);
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
      // Check if all sends have completed
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

  LOG(INFO) << "Finished graph edge discovery using NBX algorithm. Number "
               "of discovered edges "
            << other_ranks.size();

  return other_ranks;
}
//-----------------------------------------------------------------------------
namespace
{
std::pair<std::vector<int>, std::vector<int>>
transpose_src_dest(std::vector<int>& dests, std::vector<int>& dest_offsets)
{
  // List of destinations, ordered by source
  // to be transposed into a list of sources, ordered by destination

  std::vector<int> source_count(dest_offsets.size() - 1);
  for (int d : dests)
    ++source_count[d];
  std::vector<int> source_offsets = {0};
  std::partial_sum(source_count.begin(), source_count.end(),
                   std::back_inserter(source_offsets));
  std::vector<int> sources(source_offsets.back());
  for (std::size_t i = 0; i < dest_offsets.size() - 1; ++i)
  {
    for (int d = dest_offsets[i]; d < dest_offsets[i + 1]; ++d)
    {
      int dest = dests[d];
      sources[source_offsets[dest]] = i;
      source_offsets[dest]++;
    }
  }
  // Reset offsets
  source_offsets = {0};
  std::partial_sum(source_count.begin(), source_count.end(),
                   std::back_inserter(source_offsets));

  return {std::move(sources), std::move(source_offsets)};
}
} // namespace
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_gather(MPI_Comm comm,
                                         std::span<const int> in_edges_local)
{
  LOG(INFO) << "Start Graph Edge computation (Gather-Scatter): "
            << in_edges_local.size();
  int rank = dolfinx::MPI::rank(comm);
  int size = dolfinx::MPI::size(comm);
  int num_in_edges = in_edges_local.size();
  std::vector<int> in_edges_count;
  in_edges_count.reserve(1);
  if (rank == 0)
    in_edges_count.resize(size);
  MPI_Gather(&num_in_edges, 1, MPI_INT, in_edges_count.data(), 1, MPI_INT, 0,
             comm);
  std::vector<int> in_edges_offset = {0};
  std::partial_sum(in_edges_count.begin(), in_edges_count.end(),
                   std::back_inserter(in_edges_offset));

  std::vector<int> in_edges;
  in_edges.reserve(1);
  in_edges.resize(in_edges_offset.back());
  MPI_Gatherv(in_edges_local.data(), in_edges_local.size(), MPI_INT,
              in_edges.data(), in_edges_count.data(), in_edges_offset.data(),
              MPI_INT, 0, comm);

  auto [out_edges, out_edges_offset]
      = transpose_src_dest(in_edges, in_edges_offset);

  std::vector<int> out_edges_count(out_edges_offset.size() - 1);
  for (std::size_t i = 0; i < out_edges_count.size(); ++i)
    out_edges_count[i] = out_edges_offset[i + 1] - out_edges_offset[i];
  int num_out_edges;
  MPI_Scatter(out_edges_count.data(), 1, MPI_INT, &num_out_edges, 1, MPI_INT, 0,
              comm);
  std::vector<int> out_edges_local(num_out_edges);
  MPI_Scatterv(out_edges.data(), out_edges_count.data(),
               out_edges_offset.data(), MPI_INT, out_edges_local.data(),
               num_out_edges, MPI_INT, 0, comm);

  LOG(INFO) << "End Graph Edge computation (Gather-Scatter): "
            << out_edges_local.size();

  return out_edges_local;
}
//-----------------------------------------------------------------------------
std::vector<int> dolfinx::MPI::compute_graph_edges(MPI_Comm comm,
                                                   std::span<const int> edges)
{
#if defined(USE_NBX)
  return compute_graph_edges_nbx(comm, edges);
#elif defined(USE_PCX)
  return compute_graph_edges_pcx(comm, edges);
#else
  return compute_graph_edges_gather(comm, edges);
#endif
}
