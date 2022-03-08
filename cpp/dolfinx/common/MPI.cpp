// Copyright (C) 2007 Magnus Vikstrøm
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPI.h"
#include <algorithm>

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
std::vector<int> dolfinx::MPI::compute_graph_edges(MPI_Comm comm,
                                                   const std::set<int>& edges)
{
  // Send '1' to ranks that I have an edge to
  std::vector<std::uint8_t> edge_count_send(dolfinx::MPI::size(comm), 0);
  std::for_each(edges.cbegin(), edges.cend(),
                [&edge_count_send](auto e) { edge_count_send[e] = 1; });
  std::vector<std::uint8_t> edge_count_recv(edge_count_send.size());
  MPI_Alltoall(edge_count_send.data(), 1, MPI_UINT8_T, edge_count_recv.data(),
               1, MPI_UINT8_T, comm);

  // Build list of rank that had an edge to me
  std::vector<int> edges1;
  for (std::size_t i = 0; i < edge_count_recv.size(); ++i)
  {
    if (edge_count_recv[i] > 0)
      edges1.push_back(i);
  }
  return edges1;
}
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_nbx(MPI_Comm comm, const std::set<int>& edges)
// const xtl::span<const int>& edges)
{
  // Start non-blocking synchronised send
  std::vector<MPI_Request> send_requests(edges.size());
  std::byte send_buffer;
  std::vector<int> _edges(edges.begin(), edges.end());

  for (std::size_t e = 0; e < _edges.size(); ++e)
  {
    MPI_Issend(&send_buffer, 1, MPI_BYTE, _edges[e], 90, comm,
               &send_requests[e]);
  }

  // Vector to holder ranks that send data to this rank
  std::vector<int> other_ranks;

  // Start receiving
  MPI_Request barrier_request;
  bool comm_complete = false;
  bool barrier_active = false;
  while (!comm_complete)
  {
    // Check for message
    int request_pending;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, 90, comm, &request_pending, &status);

    // Check if message is waiting to be procssed
    if (request_pending)
    {
      // Receive it
      int other_rank = status.MPI_SOURCE;
      std::byte buffer_recv;
      MPI_Recv(&buffer_recv, 1, MPI_BYTE, other_rank, 90, comm,
               MPI_STATUS_IGNORE);
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

  return other_ranks;
}
//-----------------------------------------------------------------------------
std::array<std::vector<int>, 2> dolfinx::MPI::neighbors(MPI_Comm comm)
{
  int status;
  MPI_Topo_test(comm, &status);
  assert(status != MPI_UNDEFINED);

  // Get list of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);
  std::vector<int> sources(indegree), destinations(outdegree);
  MPI_Dist_graph_neighbors(comm, indegree, sources.data(), MPI_UNWEIGHTED,
                           outdegree, destinations.data(), MPI_UNWEIGHTED);

  return {std::move(sources), std::move(destinations)};
}
//-----------------------------------------------------------------------------
