// Copyright (C) 2007-2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPI.h"
#include <algorithm>
#include <dolfinx/common/log.h>

//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(MPI_Comm comm, bool duplicate)
{
  // Duplicate communicator
  if (duplicate and comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_dup(comm, &_comm);
    if (err != MPI_SUCCESS)
    {
      throw std::runtime_error(
          "Duplication of MPI communicator failed (MPI_Comm_dup)");
    }
  }
  else
    _comm = comm;
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(const Comm& comm) noexcept : Comm(comm._comm)
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
    if (err != MPI_SUCCESS)
    {
      std::cout << "Error when destroying communicator (MPI_Comm_free)."
                << std::endl;
    }
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
    if (err != MPI_SUCCESS)
    {
      std::cout << "Error when destroying communicator (MPI_Comm_free)."
                << std::endl;
    }
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
  MPI_Comm_rank(comm, &rank);
  return rank;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
int dolfinx::MPI::size(const MPI_Comm comm)
{
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_pcx(MPI_Comm comm,
                                      const xtl::span<const int>& edges)
{
  LOG(INFO)
      << "Computing communicaton graph edges (using PCX algorithm). Number "
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
  MPI_Ireduce_scatter(edge_count_send.data(), &in_edges, recvcounts.data(),
                      MPI_INT, MPI_SUM, comm, &request_scatter);

  std::vector<MPI_Request> send_requests(edges.size());
  std::byte send_buffer;
  for (std::size_t e = 0; e < edges.size(); ++e)
  {
    MPI_Isend(&send_buffer, 1, MPI_BYTE, edges[e],
              static_cast<int>(tag::consensus_pcx), comm, &send_requests[e]);
  }

  // Probe for incoming messages and store incoming rank
  MPI_Wait(&request_scatter, MPI_STATUS_IGNORE);
  std::vector<int> other_ranks;
  while (in_edges > 0)
  {
    // Check for message
    int request_pending;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, static_cast<int>(tag::consensus_pcx), comm,
               &request_pending, &status);
    if (request_pending)
    {
      // Receive message and store rank
      int other_rank = status.MPI_SOURCE;
      std::byte buffer_recv;
      MPI_Recv(&buffer_recv, 1, MPI_BYTE, other_rank,
               static_cast<int>(tag::consensus_pcx), comm, MPI_STATUS_IGNORE);
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
dolfinx::MPI::compute_graph_edges_nbx(MPI_Comm comm,
                                      const xtl::span<const int>& edges)
{
  LOG(INFO)
      << "Computing communicaton graph edges (using NBX algorithm). Number "
         "of input edges: "
      << edges.size();

  // Start non-blocking synchronised send
  std::vector<MPI_Request> send_requests(edges.size());
  std::byte send_buffer;
  for (std::size_t e = 0; e < edges.size(); ++e)
  {
    MPI_Issend(&send_buffer, 1, MPI_BYTE, edges[e],
               static_cast<int>(tag::consensus_pex), comm, &send_requests[e]);
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
    MPI_Iprobe(MPI_ANY_SOURCE, static_cast<int>(tag::consensus_pex), comm,
               &request_pending, &status);

    // Check if message is waiting to be procssed
    if (request_pending)
    {
      // Receive it
      int other_rank = status.MPI_SOURCE;
      std::byte buffer_recv;
      MPI_Recv(&buffer_recv, 1, MPI_BYTE, other_rank,
               static_cast<int>(tag::consensus_pex), comm, MPI_STATUS_IGNORE);
      other_ranks.push_back(other_rank);
    }

    if (barrier_active)
    {
      // Check for barrier completion
      int flag = 0;
      MPI_Test(&barrier_request, &flag, MPI_STATUS_IGNORE);
      if (flag)
        comm_complete = true;
    }
    else
    {
      // Check if all sends have completed
      int flag = 0;
      MPI_Testall(send_requests.size(), send_requests.data(), &flag,
                  MPI_STATUSES_IGNORE);
      if (flag)
      {
        // All send have completed, start non-blocking barrier
        MPI_Ibarrier(comm, &barrier_request);
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
