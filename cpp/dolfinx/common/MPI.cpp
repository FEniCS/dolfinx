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
  // Send '1' to ranks that I have a edge to
  std::vector<std::uint8_t> edge_count(dolfinx::MPI::size(comm), 0);
  std::for_each(edges.cbegin(), edges.cend(),
                [&edge_count](auto e) { edge_count[e] = 1; });
  MPI_Alltoall(MPI_IN_PLACE, 1, MPI_UINT8_T, edge_count.data(), 1, MPI_UINT8_T,
               comm);

  // Build list of rank that had an edge to me
  std::vector<int> edges1;
  for (std::size_t i = 0; i < edge_count.size(); ++i)
  {
    if (edge_count[i] > 0)
      edges1.push_back(i);
  }
  return edges1;
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<int>, std::vector<int>>
dolfinx::MPI::neighbors(MPI_Comm neighbor_comm)
{
  int status;
  MPI_Topo_test(neighbor_comm, &status);
  assert(status != MPI_UNDEFINED);

  // Get list of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbor_comm, &indegree, &outdegree,
                                 &weighted);
  std::vector<int> sources(indegree), destinations(outdegree);
  MPI_Dist_graph_neighbors(neighbor_comm, indegree, sources.data(),
                           MPI_UNWEIGHTED, outdegree, destinations.data(),
                           MPI_UNWEIGHTED);

  return {std::move(sources), std::move(destinations)};
}
//-----------------------------------------------------------------------------
