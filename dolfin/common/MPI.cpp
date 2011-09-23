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
#include "mpiutils.h"
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
void dolfin::MPI::distribute(std::vector<uint>& values,
                             std::vector<uint>& partition)
{
  dolfin::distribute(values, partition);
}
//-----------------------------------------------------------------------------
void dolfin::MPI::distribute(std::vector<int>& values,
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
void dolfin::MPI::distribute(std::vector<bool>& values,
                             std::vector<uint>& partition)
{
  error("MPI::distribute does not yet support bool. It needs to be manage as a special case."); 
}
//-----------------------------------------------------------------------------
void dolfin::MPI::scatter(std::vector<uint>& values, uint sending_process)
{
  // Prepare receive buffer (size 1)
  int receive_buffer = 0;

  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Prepare arguments differently depending on whether we're sending
  if (process_number() == sending_process)
  {
    // Check size of values
    if (values.size() != num_processes())
      dolfin_error("MPI.cpp",
                   "scatter values across processes",
                   "The number of values (%d) does not match the number of processes (%d)",
                   values.size(), num_processes());

    // Prepare send buffer
    uint* send_buffer = new uint[values.size()];
    for (uint i = 0; i < values.size(); i++)
      send_buffer[i] = values[i];

    // Call MPI to send values
    MPI_Scatter(send_buffer,
                1,
                MPI_UNSIGNED,
                &receive_buffer,
                1,
                MPI_UNSIGNED,
                static_cast<int>(sending_process),
                *comm);

    // Cleanup
    delete [] send_buffer;
  }
  else
  {
    // Call MPI to receive values
    MPI_Scatter(0,
                0,
                MPI_UNSIGNED,
                &receive_buffer,
                1,
                MPI_UNSIGNED,
                static_cast<int>(sending_process),
                *comm);
  }

  // Collect values
  values.clear();
  values.push_back(receive_buffer);
}
//-----------------------------------------------------------------------------
void dolfin::MPI::scatter(std::vector<std::vector<uint> >& values,
                          uint sending_process)
{
  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Receive buffer
  int recv_count = 0;
  uint* recv_buffer = 0;

  // Prepare arguments differently depending on whether we're sending
  if (process_number() == sending_process)
  {
    // Check size of values
    if (values.size() != num_processes())
      dolfin_error("MPI.cpp",
                   "scatter values across processes",
                   "The number of values (%d) does not match the number of processes (%d)",
                   values.size(), num_processes());

    // Extract sizes and compute size of send buffer
    std::vector<uint> sizes;
    for (uint i = 0; i < values.size(); ++i)
      sizes.push_back(values[i].size());
    int send_buffer_size = std::accumulate(sizes.begin(), sizes.end(), 0);

    // Build send data
    uint* send_buffer = new uint[send_buffer_size];
    int* send_counts = new int[values.size()];
    int* send_offsets = new int[values.size()];
    send_offsets[0] = 0;
    uint offset = 0;
    for (uint i = 0; i < values.size(); ++i)
    {
      send_counts[i] = sizes[i];
      send_offsets[i] = offset;
      for (uint j = 0; j < values[i].size(); ++j)
        send_buffer[offset++] = values[i][j];
    }

    // Scatter number of values that will be scattered (note that sizes will be modified)
    scatter(sizes, sending_process);

    // Prepare receive buffer
    recv_count = sizes[0];
    recv_buffer = new uint[recv_count];

    // Call MPI to send values
    MPI_Scatterv(send_buffer,
                 send_counts,
                 send_offsets,
                 MPI_UNSIGNED,
                 recv_buffer,
                 recv_count,
                 MPI_UNSIGNED,
                 static_cast<int>(sending_process),
                 *comm);

    // Cleanup
    delete [] send_buffer;
    delete [] send_counts;
    delete [] send_offsets;
  }
  else
  {
    // Receive number of values that will be scattered
    std::vector<uint> sizes;
    scatter(sizes, sending_process);

    // Prepare receive buffer
    recv_count = sizes[0];
    recv_buffer = new uint[recv_count];

    // Call MPI to receive values
    MPI_Scatterv(0,
                 0,
                 0,
                 MPI_UNSIGNED,
                 recv_buffer,
                 recv_count,
                 MPI_UNSIGNED,
                 static_cast<int>(sending_process),
                 *comm);
  }

  // Copy values from receive buffer
  values.clear();
  values.resize(1);
  for (int i = 0; i < recv_count; ++i)
    values[0].push_back(recv_buffer[i]);

  // Cleanup
  delete [] recv_buffer;
}
//-----------------------------------------------------------------------------
void dolfin::MPI::scatter(std::vector<std::vector<int> >& values,
                          uint sending_process)
{
  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Receive buffer
  int recv_count = 0;
  int* recv_buffer = 0;

  // Prepare arguments differently depending on whether we're sending
  if (process_number() == sending_process)
  {
    // Check size of values
    if (values.size() != num_processes())
      dolfin_error("MPI.cpp",
                   "scatter values across processes",
                   "The number of values (%d) does not match the number of processes (%d)",
                   values.size(), num_processes());

    // Extract sizes and compute size of send buffer
    std::vector<uint> sizes;
    for (uint i = 0; i < values.size(); ++i)
      sizes.push_back(values[i].size());
    int send_buffer_size = std::accumulate(sizes.begin(), sizes.end(), 0);

    // Build send data
    int* send_buffer = new int[send_buffer_size];
    int* send_counts = new int[values.size()];
    int* send_offsets = new int[values.size()];
    send_offsets[0] = 0;
    uint offset = 0;
    for (uint i = 0; i < values.size(); ++i)
    {
      send_counts[i] = sizes[i];
      send_offsets[i] = offset;
      for (uint j = 0; j < values[i].size(); ++j)
        send_buffer[offset++] = values[i][j];
    }

    // Scatter number of values that will be scattered (note that sizes will be modified)
    scatter(sizes, sending_process);

    // Prepare receive buffer
    recv_count = sizes[0];
    recv_buffer = new int[recv_count];

    // Call MPI to send values
    MPI_Scatterv(send_buffer,
                 send_counts,
                 send_offsets,
                 MPI_UNSIGNED,
                 recv_buffer,
                 recv_count,
                 MPI_UNSIGNED,
                 static_cast<int>(sending_process),
                 *comm);

    // Cleanup
    delete [] send_buffer;
    delete [] send_counts;
    delete [] send_offsets;
  }
  else
  {
    // Receive number of values that will be scattered
    std::vector<uint> sizes;
    scatter(sizes, sending_process);

    // Prepare receive buffer
    recv_count = sizes[0];
    recv_buffer = new int[recv_count];

    // Call MPI to receive values
    MPI_Scatterv(0,
                 0,
                 0,
                 MPI_INT,
                 recv_buffer,
                 recv_count,
                 MPI_INT,
                 static_cast<int>(sending_process),
                 *comm);
  }

  // Copy values from receive buffer
  values.clear();
  values.resize(1);
  for (int i = 0; i < recv_count; ++i)
    values[0].push_back(recv_buffer[i]);

  // Cleanup
  delete [] recv_buffer;
}
//-----------------------------------------------------------------------------
void dolfin::MPI::scatter(std::vector<std::vector<double> >& values,
                          uint sending_process)
{
  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Receive buffer
  int recv_count = 0;
  double* recv_buffer = 0;

  // Prepare arguments differently depending on whether we're sending
  if (process_number() == sending_process)
  {
    // Check size of values
    if (values.size() != num_processes())
      dolfin_error("MPI.cpp",
                   "scatter values across processes",
                   "The number of values (%d) does not match the number of processes (%d)",
                   values.size(), num_processes());

    // Extract sizes and compute size of send buffer
    std::vector<uint> sizes;
    for (uint i = 0; i < values.size(); ++i)
      sizes.push_back(values[i].size());
    int send_buffer_size = std::accumulate(sizes.begin(), sizes.end(), 0);

    // Build send data
    double* send_buffer = new double[send_buffer_size];
    int* send_counts = new int[values.size()];
    int* send_offsets = new int[values.size()];
    send_offsets[0] = 0;
    uint offset = 0;
    for (uint i = 0; i < values.size(); ++i)
    {
      send_counts[i] = sizes[i];
      send_offsets[i] = offset;
      for (uint j = 0; j < values[i].size(); ++j)
        send_buffer[offset++] = values[i][j];
    }

    // Scatter number of values that will be scattered (note that sizes will be modified)
    scatter(sizes, sending_process);

    // Prepare receive buffer
    recv_count = sizes[0];
    recv_buffer = new double[recv_count];

    // Call MPI to send values
    MPI_Scatterv(send_buffer,
                 send_counts,
                 send_offsets,
                 MPI_DOUBLE,
                 recv_buffer,
                 recv_count,
                 MPI_DOUBLE,
                 static_cast<int>(sending_process),
                 *comm);

    // Cleanup
    delete [] send_buffer;
    delete [] send_counts;
    delete [] send_offsets;
  }
  else
  {
    // Receive number of values that will be scattered
    std::vector<uint> sizes;
    scatter(sizes, sending_process);

    // Prepare receive buffer
    recv_count = sizes[0];
    recv_buffer = new double[recv_count];

    // Call MPI to receive values
    MPI_Scatterv(0,
                 0,
                 0,
                 MPI_DOUBLE,
                 recv_buffer,
                 recv_count,
                 MPI_DOUBLE,
                 static_cast<int>(sending_process),
                 *comm);
  }

  // Copy values from receive buffer
  values.clear();
  values.resize(1);
  for (int i = 0; i < recv_count; ++i)
    values[0].push_back(recv_buffer[i]);

  // Cleanup
  delete [] recv_buffer;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> dolfin::MPI::gather(uint value)
{
  std::vector<uint> values(num_processes());
  values[process_number()] = value;
  gather(values);
  return values;
}
//-----------------------------------------------------------------------------
/*
dolfin::uint dolfin::MPI::global_maximum(uint size)
{
  uint recv_size = 0;
  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;
  MPI_Allreduce(&size, &recv_size, 1, MPI_UNSIGNED, MPI_MAX, *comm);
  return recv_size;
}
*/
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::global_offset(uint range, bool exclusive)
{
  uint offset = 0;

  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Compute inclusive or exclusive partial reduction
  if (exclusive)
    MPI_Exscan(&range, &offset, 1, MPI_UNSIGNED, MPI_SUM, *comm);
  else
    MPI_Scan(&range, &offset, 1, MPI_UNSIGNED, MPI_SUM, *comm);

  return offset;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(uint* send_buffer, uint send_size, uint dest,
                                    uint* recv_buffer, uint recv_size, uint source)
{
  MPI_Status status;

  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Send and receive data
  MPI_Sendrecv(send_buffer, static_cast<int>(send_size), MPI_UNSIGNED, static_cast<int>(dest), 0,
               recv_buffer, static_cast<int>(recv_size), MPI_UNSIGNED, static_cast<int>(source),  0,
               *comm, &status);

  // Check number of received values
  int num_received = 0;
  MPI_Get_count(&status, MPI_UNSIGNED, &num_received);
  assert(num_received >= 0);

  return static_cast<uint>(num_received);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(int* send_buffer, uint send_size, uint dest,
                                    int* recv_buffer, uint recv_size, uint source)
{
  MPI_Status status;

  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Send and receive data
  MPI_Sendrecv(send_buffer, static_cast<int>(send_size), MPI_INT, static_cast<int>(dest), 0,
               recv_buffer, static_cast<int>(recv_size), MPI_INT, static_cast<int>(source),  0,
               *comm, &status);

  // Check number of received values
  int num_received = 0;
  MPI_Get_count(&status, MPI_INT, &num_received);
  assert(num_received >= 0);

  return static_cast<uint>(num_received);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(double* send_buffer, uint send_size, uint dest,
                                    double* recv_buffer, uint recv_size, uint source)
{
  MPI_Status status;

  // Create communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Send and receive data
  MPI_Sendrecv(send_buffer, static_cast<int>(send_size), MPI_DOUBLE, static_cast<int>(dest), 0,
               recv_buffer, static_cast<int>(recv_size), MPI_DOUBLE, static_cast<int>(source),  0,
               *comm, &status);

  // Check number of received values
  int num_received = 0;
  MPI_Get_count(&status, MPI_DOUBLE, &num_received);
  assert(num_received >= 0);

  return static_cast<uint>(num_received);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(bool* send_buffer, uint send_size, uint dest,
                                    bool* recv_buffer, uint recv_size, uint source)
{
  error("MPI::send_recv does not yet support bool. It needs to be manage as a special case."); 
  return 0;
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
  assert(index < N);

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
void dolfin::MPI::distribute(std::vector<uint>& values,
                             std::vector<uint>& partition)
{
  dolfin_error("MPI.cpp",
               "call MPI::distribute",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::distribute(std::vector<int>& values,
                             std::vector<uint>& partition)
{
  dolfin_error("MPI.cpp",
               "call MPI::distribute",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::distribute(std::vector<double>& values,
                             std::vector<uint>& partition)
{
  dolfin_error("MPI.cpp",
               "call MPI::distribute",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::distribute(std::vector<bool>& values,
                             std::vector<uint>& partition)
{
  dolfin_error("MPI.cpp",
               "call MPI::distribute",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::scatter(std::vector<uint>& values, uint sending_process)
{
  dolfin_error("MPI.cpp",
               "call MPI::scatter",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::scatter(std::vector<std::vector<uint> >& values,
                          uint sending_process)
{
  dolfin_error("MPI.cpp",
               "call MPI::scatter",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::scatter(std::vector<std::vector<int> >& values,
                          uint sending_process)
{
  dolfin_error("MPI.cpp",
               "call MPI::scatter",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
void dolfin::MPI::scatter(std::vector<std::vector<double> >& values,
                          uint sending_process)
{
  dolfin_error("MPI.cpp",
               "call MPI::scatter",
               "Your DOLFIN installation has been built without MPI support");
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> dolfin::MPI::gather(uint value)
{
  dolfin_error("MPI.cpp",
               "call MPI::gather",
               "Your DOLFIN installation has been built without MPI support");
  return std::vector<uint>(1);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::global_offset(uint range, bool exclusive)
{
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(uint* send_buffer, uint send_size, uint dest,
                                    uint* recv_buffer, uint recv_size, uint source)
{
  dolfin_error("MPI.cpp",
               "call MPI::send_recv",
               "Your DOLFIN installation has been built without MPI support");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(int* send_buffer, uint send_size, uint dest,
                                    int* recv_buffer, uint recv_size, uint source)
{
  dolfin_error("MPI.cpp",
               "call MPI::send_recv",
               "Your DOLFIN installation has been built without MPI support");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(double* send_buffer, uint send_size, uint dest,
                                    double* recv_buffer, uint recv_size, uint source)
{
  dolfin_error("MPI.cpp",
               "call MPI::send_recv",
               "Your DOLFIN installation has been built without MPI support");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::send_recv(bool* send_buffer, uint send_size, uint dest,
                                    bool* recv_buffer, uint recv_size, uint source)
{
  dolfin_error("MPI.cpp",
               "call MPI::send_recv",
               "Your DOLFIN installation has been built without MPI support");
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
    error("MPI is required for local_range with more than one process.");
  return std::make_pair(0, N);
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> dolfin::MPI::local_range(uint process,
                                                               uint N,
                                                               uint num_processes)
{
  if (process != 0 || num_processes > 1)
    error("MPI is required for local_range with more than one process.");
  return std::make_pair(0, N);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::index_owner(uint i, uint N)
{
  assert(i < N);
  return 0;
}
//-----------------------------------------------------------------------------
#endif
