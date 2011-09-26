// Copyright (C) 2008-2009 Niclas Jansson, Anders Logg and Ola Skavhaug
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
// Credits to Scott Baden for suggesting the algorithm
//
// First added:  2008-12-09
// Last changed: 2009-06-29
//
// Template utilities for MPI class placed here so it will not clutter
// the MPI class.

#ifndef __MPI_UTILS_H
#define __MPI_UTILS_H

#include <vector>
#include <dolfin/log/log.h>
#include <dolfin/common/types.h>
#include "MPI.h"

#ifdef HAS_MPI

#include <mpi.h>

namespace dolfin
{

  /// Distribute local arrays on all processors according to given partition
  template <typename T>
  void distribute(std::vector<T>& values, std::vector<uint>& partition)
  {
    assert(values.size() == partition.size());

    // Get number of processes and process number
    const uint num_processes  = MPI::num_processes();
    const uint process_number = MPI::process_number();

    // Sort out data that should be sent to other processes
    std::vector<std::vector<T> > send_data(num_processes);
    for (uint i = 0; i < values.size(); i++)
    {
      // Get process number data should be sent to
      const uint p = partition[i];
      assert(p < send_data.size());

      // Append data to array for process p
      send_data[p].push_back(values[i]);
    }

    // Store local data (don't send) and clear partition vector and reuse for
    // storing sender of data
    values.clear();
    partition.clear();
    const std::vector<T>& local_values = send_data[process_number];
    for (uint i = 0; i < local_values.size(); i++)
    {
      values.push_back(local_values[i]);
      partition.push_back(process_number);
    }

    // Determine size of send buffer
    uint send_buffer_size = 0;
    for (uint p = 0; p < send_data.size(); p++)
    {
      if (p == process_number)
        continue;
      send_buffer_size = std::max(send_buffer_size, static_cast<uint>(send_data[p].size()));
    }

    // Determine size of receive buffer (max. values across for all processes,
    // make receive buffer size same on all processes)
    uint recv_buffer_size = 0;
    MPI_Allreduce(&send_buffer_size, &recv_buffer_size, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);

    // Allocate memory for send and receive buffers
    // assert(send_buffer_size > 0);
    // assert(recv_buffer_size > 0);
    std::vector<T> send_buffer(send_buffer_size + 1);
    std::vector<T> recv_buffer(recv_buffer_size + 1);

    // Exchange data
    for (uint i = 1; i < send_data.size(); i++)
    {
      // We receive data from process p - i (i steps to the left)
      const int source = (process_number - i + num_processes) % num_processes;

      // We send data to process p + i (i steps to the right)
      const int dest = (process_number + i) % num_processes;

      // Copy data to send buffer
      for (uint j = 0; j < send_data[dest].size(); j++)
        send_buffer[j] = send_data[dest][j];

      // Send and receive data
      const uint num_received = MPI::send_recv(&send_buffer[0], send_data[dest].size(), dest,
                                               &recv_buffer[0], recv_buffer_size,       source);

      // Copy data from receive buffer
      assert(num_received <= recv_buffer_size);
      for (uint j = 0; j < num_received; j++)
      {
        values.push_back(recv_buffer[j]);
        partition.push_back(source);
      }
    }
  }

}

#else

namespace dolfin
{

  /// Distribute local arrays on all processors according to given partition
  template <typename T>
  void distribute(std::vector<T>& values, const std::vector<uint> partition)
  {
    error("Distribution of partitioned values requires MPI.");
  }

}

#endif

#endif
