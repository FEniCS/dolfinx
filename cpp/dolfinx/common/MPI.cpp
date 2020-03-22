// Copyright (C) 2007 Magnus Vikstrøm
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPI.h"
#include "SubSystemsManager.h"
#include <algorithm>
#include <numeric>

//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(MPI_Comm comm)
{
  // Duplicate communicator
  if (comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_dup(comm, &_comm);
    if (err != MPI_SUCCESS)
    {
      throw std::runtime_error(
          "Duplication of MPI communicator failed (MPI_Comm_dup)");
    }
  }
  else
    _comm = MPI_COMM_NULL;
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(const Comm& comm) : Comm(comm._comm)
{
  // FIXME: should probably duplicate the comm here
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(Comm&& comm)
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
dolfinx::MPI::Comm::operator=(dolfinx::MPI::Comm&& comm)
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
  // Return
  return *this;
}
//-----------------------------------------------------------------------------
MPI_Comm dolfinx::MPI::Comm::comm() const { return _comm; }
//-----------------------------------------------------------------------------
std::uint32_t dolfinx::MPI::rank(const MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}
//-----------------------------------------------------------------------------
std::uint32_t dolfinx::MPI::size(const MPI_Comm comm)
{
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}
//-----------------------------------------------------------------------------
void dolfinx::MPI::barrier(const MPI_Comm comm) { MPI_Barrier(comm); }
//-----------------------------------------------------------------------------
std::size_t dolfinx::MPI::global_offset(const MPI_Comm comm, std::size_t range,
                                        bool exclusive)
{
  // Compute inclusive or exclusive partial reduction
  std::size_t offset = 0;
  MPI_Scan(&range, &offset, 1, mpi_type<std::size_t>(), MPI_SUM, comm);
  if (exclusive)
    offset -= range;
  return offset;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> dolfinx::MPI::local_range(const MPI_Comm comm,
                                                      std::int64_t N)
{
  return local_range(comm, rank(comm), N);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2>
dolfinx::MPI::local_range(const MPI_Comm comm, int process, std::int64_t N)
{
  return compute_local_range(process, N, size(comm));
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2>
dolfinx::MPI::compute_local_range(int process, std::int64_t N, int size)
{
  assert(process >= 0);
  assert(N >= 0);
  assert(size > 0);

  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  if (process < r)
    return {{process * (n + 1), process * (n + 1) + n + 1}};
  else
    return {{process * n + r, process * n + r + n}};
}
//-----------------------------------------------------------------------------
std::uint32_t dolfinx::MPI::index_owner(int size, std::size_t index,
                                        std::size_t N)
{
  assert(index < N);

  // Compute number of items per process and remainder
  const std::size_t n = N / size;
  const std::size_t r = N % size;

  // First r processes own n + 1 indices
  if (index < r * (n + 1))
    return index / (n + 1);

  // Remaining processes own n indices
  return r + (index - r * (n + 1)) / n;
}
//-----------------------------------------------------------------------------
MPI_Comm dolfinx::MPI::SubsetComm(MPI_Comm comm, int num_processes)
{

  int comm_size = MPI::size(comm);
  MPI_Comm new_comm;
  if (num_processes <= 0)
  {

    throw std::runtime_error(
        "The sub-communicator should be composed by at least one process.");
  }
  else if (comm_size < num_processes)
  {

    throw std::runtime_error("Cannot create a sub-communicator with more "
                             "processes than the original communicator.");
  }
  else if (comm_size == num_processes)
  {
    // Make a copy of the orginal communicator
    int err = MPI_Comm_dup(comm, &new_comm);
    if (err != MPI_SUCCESS)
    {
      throw std::runtime_error(
          "Duplication of MPI communicator failed (MPI_Comm_dup)");
    }
    return new_comm;
  }
  else
  {
    // Get the group of all processes in comm
    MPI_Group comm_group;
    MPI_Comm_group(comm, &comm_group);

    // Select N processes to compose new communicator
    // TODO: Could find a better subset of processors?
    std::vector<int> ranks(num_processes);
    std::iota(ranks.begin(), ranks.end(), 0);

    // Construct a group containing num_processes first processes
    MPI_Group new_group;
    MPI_Group_incl(comm_group, num_processes, ranks.data(), &new_group);

    // Create a new communicator based on the group new group
    int err = MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

    if (err != MPI_SUCCESS)
    {
      throw std::runtime_error(
          "Creation of a new MPI communicator failed (MPI_Comm_create_group)");
    }

    // free groups
    MPI_Group_free(&comm_group);
    MPI_Group_free(&new_group);

    return new_comm;
  }
}
//-----------------------------------------------------------------------------
std::vector<int> dolfinx::MPI::neighbors(MPI_Comm neighbor_comm)
{
  // Get list of neighbours
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbor_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  std::vector<int> neighbors(indegree), neighbors1(indegree);

  MPI_Dist_graph_neighbors(neighbor_comm, indegree, neighbors.data(),
                           MPI_UNWEIGHTED, outdegree, neighbors1.data(),
                           MPI_UNWEIGHTED);
  return neighbors;
}
//-----------------------------------------------------------------------------
