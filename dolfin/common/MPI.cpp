// Copyright (C) 2007 Magnus Vikstrøm
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2007-20!2
// Modified by Anders Logg 2007-2011
// Modified by Ola Skavhaug 2008-2009
// Modified by Niclas Jansson 2009
// Modified by Joachim B Haga 2012
//
// First added:  2007-11-30
// Last changed: 2012-11-17

#include <numeric>
#include <dolfin/log/dolfin_log.h>
#include "SubSystemsManager.h"
#include "MPI.h"

#ifdef HAS_MPI

//-----------------------------------------------------------------------------
dolfin::MPIInfo::MPIInfo()
{
  MPI_Info_create(&info);
}
//-----------------------------------------------------------------------------
dolfin::MPIInfo::~MPIInfo()
{
  MPI_Info_free(&info);
}
//-----------------------------------------------------------------------------
MPI_Info& dolfin::MPIInfo::operator*()
{
  return info;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void dolfin::MPINonblocking::wait_all()
{
  if (!reqs.empty())
  {
    boost::mpi::wait_all(reqs.begin(), reqs.end());
    reqs.clear();
  }
}
#endif
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
unsigned int dolfin::MPI::process_number()
{
  deprecation("MPI::process_number",
              "1.4", "1.5",
              "MPI::process_number() has been replaced by MPI::rank(MPI_Comm).");

#ifdef HAS_MPI
  SubSystemsManager::init_mpi();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}
//-----------------------------------------------------------------------------
unsigned int dolfin::MPI::num_processes()
{
  deprecation("MPI::num_processes",
              "1.4", "1.5",
              "MPI::num_processes() has been replaced by MPI::size(MPI_Comm).");

#ifdef HAS_MPI
  SubSystemsManager::init_mpi();
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
#else
  return 1;
#endif
}
//-----------------------------------------------------------------------------
unsigned int dolfin::MPI::rank(const MPI_Comm comm)
{
#ifdef HAS_MPI
  SubSystemsManager::init_mpi();
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
#else
  return 0;
#endif
}
//-----------------------------------------------------------------------------
unsigned int dolfin::MPI::size(const MPI_Comm comm)
{
#ifdef HAS_MPI
  SubSystemsManager::init_mpi();
  int size;
  MPI_Comm_size(comm, &size);
  return size;
#else
  return 1;
#endif
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::is_broadcaster(const MPI_Comm comm)
{
  // Always broadcast from processor number 0
  return size(comm) > 1 && rank(comm) == 0;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::is_receiver(const MPI_Comm comm)
{
  // Always receive on processors with numbers > 0
  return size(comm) > 1 && rank(comm) > 0;
}
//-----------------------------------------------------------------------------
void dolfin::MPI::barrier(const MPI_Comm comm)
{
#ifdef HAS_MPI
  MPI_Barrier(comm);
#endif
}
//-----------------------------------------------------------------------------
std::size_t dolfin::MPI::global_offset(const MPI_Comm comm,
                                       std::size_t range, bool exclusive)
{
#ifdef HAS_MPI
  // Compute inclusive or exclusive partial reduction
  std::size_t offset = 0;
  MPI_Scan(&range, &offset, 1, mpi_type<std::size_t>(), MPI_SUM, comm);
  if (exclusive)
    offset -= range;
  return offset;
#else
  return 0;
#endif
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
dolfin::MPI::local_range(const MPI_Comm comm, std::size_t N)
{
  return local_range(comm, rank(comm), N);
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
dolfin::MPI::local_range(const MPI_Comm comm, unsigned int process,
                         std::size_t N)
{
  return compute_local_range(process, N, size(comm));
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
dolfin::MPI::compute_local_range(unsigned int process,
                                 std::size_t N,
                                 unsigned int size)
{
  // Compute number of items per process and remainder
  const std::size_t n = N / size;
  const std::size_t r = N % size;

  // Compute local range
  std::pair<std::size_t, std::size_t> range;
  if (process < r)
  {
    range.first = process*(n + 1);
    range.second = range.first + n + 1;
  }
  else
  {
    range.first = process*n + r;
    range.second = range.first + n;
  }

  return range;
}
//-----------------------------------------------------------------------------
unsigned int dolfin::MPI::index_owner(const MPI_Comm comm,
                                      std::size_t index, std::size_t N)
{
  dolfin_assert(index < N);

  // Get number of processes
  const unsigned int _size = size(comm);

  // Compute number of items per process and remainder
  const std::size_t n = N / _size;
  const std::size_t r = N % _size;

  // First r processes own n + 1 indices
  if (index < r * (n + 1))
    return index / (n + 1);

  // Remaining processes own n indices
  return r + (index - r * (n + 1)) / n;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
#ifndef HAS_MPI
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void dolfin::MPINonblocking::wait_all()
{
  dolfin_error("MPI.h",
               "call MPINonblocking::wait_all",
               "DOLFIN has been configured without MPI support");
}
//-----------------------------------------------------------------------------
#endif
