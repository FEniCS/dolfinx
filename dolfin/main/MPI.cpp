// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007, 2008.
// Modified by Anders Logg, 2007-2009.
// Modified by Ola Skavhaug, 2008-2009.
//
// First added:  2007-11-30
// Last changed: 2009-05-25

#include "mpiutils.h"
#include "SubSystemsManager.h"
#include "MPI.h"

#ifdef HAS_MPI

#include <mpi.h>

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
  SubSystemsManager::initMPI();
  return static_cast<uint>(COMM_WORLD.Get_rank());
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::num_processes()
{
  SubSystemsManager::initMPI();
  return static_cast<uint>(COMM_WORLD.Get_size());
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::broadcast()
{
  // Always broadcast from processor number 0
  return num_processes() > 1 && process_number() == 0;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::receive()
{
  // Always receive on processors with numbers > 0
  return num_processes() > 1 && process_number() > 0;
}
//-----------------------------------------------------------------------------
void dolfin::MPI::distribute(std::vector<uint>& values,
                             std::vector<uint>& partition)
{
  dolfin::distribute(values, partition);
}
//-----------------------------------------------------------------------------
void dolfin::MPI::distribute(std::vector<double>& values,
                             std::vector<uint>& partition)
{
  dolfin::distribute(values, partition);
}
//-----------------------------------------------------------------------------
void dolfin::MPI::gather(std::vector<uint>& values)
{
  dolfin_assert(values.size() == num_processes());

  // Prepare arrays
  uint send_value = values[process_number()];
  uint* received_values = new uint[values.size()];

  /// Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;
  // Call MPI
  MPI_Allgather(&send_value,     1, MPI_UNSIGNED,
                received_values, 1, MPI_UNSIGNED, *comm);

  // Copy values
  for (uint i = 0; i < values.size(); i++)
    values[i] = received_values[i];

  // Cleanup
  delete [] received_values;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::global_maximum(uint size)
{
  uint recv_size = 0;
  /// Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;
  MPI_Allreduce(&size, &recv_size, 1, MPI_UNSIGNED, MPI_MAX, *comm);
  return recv_size;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(uint* send_buffer, uint send_size, uint dest,
                                    uint* recv_buffer, uint recv_size, uint source)
{
  MPI_Status status;

  dolfin_debug2("Sending to %d, receiving from %d", dest, source);

  /// Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Send and receive data
  MPI_Sendrecv(send_buffer, static_cast<int>(send_size), MPI_UNSIGNED, static_cast<int>(dest), 0,
               recv_buffer, static_cast<int>(recv_size), MPI_UNSIGNED, static_cast<int>(source),  0,
               *comm, &status);

  // Check number of received values
  int num_received = 0;
  MPI_Get_count(&status, MPI_UNSIGNED, &num_received);
  dolfin_assert(num_received >= 0);
  dolfin_debug1("Received %d values", num_received);

  return static_cast<uint>(num_received);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(double* send_buffer, uint send_size, uint dest,
                                    double* recv_buffer, uint recv_size, uint source)
{
  MPI_Status status;

  /// Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Send and receive data
  MPI_Sendrecv(send_buffer, static_cast<int>(send_size), MPI_DOUBLE, static_cast<int>(dest), 0,
               recv_buffer, static_cast<int>(recv_size), MPI_DOUBLE, static_cast<int>(source),  0,
               *comm, &status);

  // Check number of received values
  int num_received = 0;
  MPI_Get_count(&status, MPI_DOUBLE, &num_received);
  dolfin_assert(num_received >= 0);

  return static_cast<uint>(num_received);
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint N)
{
  // Get number of processes and process number
  const uint _num_processes = num_processes();
  const uint _process_number = process_number();

  // Compute number of items per process and remainder
  const uint n = N / _num_processes;
  const uint r = N % _num_processes;

  // Compute local range
  std::pair<uint, uint> range;
  if (_process_number < r)
  {
    range.first = _process_number*(n + 1);
    range.second = range.first + n + 1;
  }
  else
  {
    range.first = _process_number*n + r;
    range.second = range.first + n;
  }

  return range;
}
//-----------------------------------------------------------------------------

#else

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
bool dolfin::MPI::broadcast()
{
  return false;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::receive()
{
  return false;
}
//-----------------------------------------------------------------------------
void dolfin::MPI::distribute(std::vector<uint>& values,
                             std::vector<uint>& partition)
{
  error("MPI::distribute() requires MPI.");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::distribute(std::vector<double>& values,
                             std::vector<uint>& partition)
{
  error("MPI::distribute() requires MPI.");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::gather(std::vector<uint>& values)
{
  error("MPI::gather() requires MPI.");
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::global_maximum(uint size)
{
  error("MPI::global_maximum() requires MPI.");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(uint* send_buffer, uint send_size, uint dest,
                                    uint* recv_buffer, uint recv_size, uint source)
{
  error("MPI::send_recv() requires MPI.");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(double* send_buffer, uint send_size, uint dest,
                                    double* recv_buffer, uint recv_size, uint source)
{
  error("MPI::send_recv() requires MPI.");
  return 0;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint N)
{
  return std::make_pair(0, N);
}
//-----------------------------------------------------------------------------

#endif
