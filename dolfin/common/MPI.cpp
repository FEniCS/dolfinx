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
// Modified by Garth N. Wells 2007-2009
// Modified by Anders Logg 2007-2011
// Modified by Ola Skavhaug 2008-2009
// Modified by Niclas Jansson 2009
//
// First added:  2007-11-30
// Last changed: 2011-08-25

#include <numeric>
#include <dolfin/log/dolfin_log.h>
#include "SubSystemsManager.h"
#include "MPI.h"

#ifdef HAS_MPI

using MPI::COMM_WORLD;

//-----------------------------------------------------------------------------
dolfin::MPICommunicator::MPICommunicator()
{
  MPI_Comm_dup(MPI_COMM_WORLD, &communicator);
}
//-----------------------------------------------------------------------------
dolfin::MPICommunicator::~MPICommunicator()
{
  MPI_Comm_free(&communicator);
}
//-----------------------------------------------------------------------------
MPI_Comm& dolfin::MPICommunicator::operator*()
{
  return communicator;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::process_number()
{
  SubSystemsManager::init_mpi();
  return static_cast<uint>(COMM_WORLD.Get_rank());
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::num_processes()
{
  SubSystemsManager::init_mpi();
  return static_cast<uint>(COMM_WORLD.Get_size());
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::is_broadcaster()
{
  // Always broadcast from processor number 0
  return num_processes() > 1 && process_number() == 0;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::is_receiver()
{
  // Always receive on processors with numbers > 0
  return num_processes() > 1 && process_number() > 0;
}
//-----------------------------------------------------------------------------
void dolfin::MPI::barrier()
{
  MPICommunicator comm;
  MPI_Barrier(*comm);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::global_offset(uint range, bool exclusive)
{
  MPICommunicator mpi_comm;
  boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_duplicate);

  // Compute inclusive or exclusive partial reduction
  dolfin::uint offset = boost::mpi::scan(comm, range, std::plus<dolfin::uint>());
  if (exclusive)
    offset -= range;

  return offset;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint N)
{
  return local_range(process_number(), N);
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint process,
                                                               uint N)
{
  return local_range(process, N, num_processes());
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint process,
                                                               uint N,
                                                               uint num_processes)
{
  // Compute number of items per process and remainder
  const uint n = N / num_processes;
  const uint r = N % num_processes;

  // Compute local range
  std::pair<uint, uint> range;
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
dolfin::uint dolfin::MPI::index_owner(uint index, uint N)
{
  dolfin_assert(index < N);

  // Get number of processes
  const uint _num_processes = num_processes();

  // Compute number of items per process and remainder
  const uint n = N / _num_processes;
  const uint r = N % _num_processes;

  // First r processes own n + 1 indices
  if (index < r * (n + 1))
    return index / (n + 1);

  // Remaining processes own n indices
  return r + (index - r * (n + 1)) / n;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
#else
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::process_number()
{
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::num_processes()
{
  return 1;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::is_broadcaster()
{
  return false;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::is_receiver()
{
  return false;
}
//-----------------------------------------------------------------------------
void dolfin::MPI::barrier()
{
  dolfin_error("MPI.cpp",
               "call MPI::barrier",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::global_offset(uint range, bool exclusive)
{
  return 0;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint N)
{
  return std::make_pair(0, N);
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint process,
                                                               uint N)
{
  if (process != 0 || num_processes() > 1)
  {
    dolfin_error("MPI.cpp",
                 "access local range for process",
                 "DOLFIN has not been configured with MPI support");
  }
  return std::make_pair(0, N);
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint process,
                                                               uint N,
                                                               uint num_processes)
{
  if (process != 0 || num_processes > 1)
  {
    dolfin_error("MPI.cpp",
                 "access local range for process",
                 "DOLFIN has not been configured with MPI support");
  }
  return std::make_pair(0, N);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::index_owner(uint i, uint N)
{
  dolfin_assert(i < N);
  return 0;
}
//-----------------------------------------------------------------------------
#endif
