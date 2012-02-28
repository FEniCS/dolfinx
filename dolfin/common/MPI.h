// Copyright (C) 2007-2011 Magnus Vikstrøm and Garth N. Wells
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
// Modified by Ola Skavhaug 2008-2009
// Modified by Anders Logg 2008-2011
// Modified by Niclas Jansson 2009
//
// First added:  2007-11-30
// Last changed: 2011-09-28

#ifndef __MPI_DOLFIN_WRAPPER_H
#define __MPI_DOLFIN_WRAPPER_H

#include <utility>
#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>

#ifdef HAS_MPI
#include <boost/mpi.hpp>
#include <mpi.h>
#endif

namespace dolfin
{

  #ifdef HAS_MPI
  class MPICommunicator
  {

  public:

    /// Create communicator (copy of MPI_COMM_WORLD)
    MPICommunicator(MPI_Comm communicator=MPI_COMM_WORLD);

    /// Destructor
    ~MPICommunicator();

    /// Dereference operator
    MPI_Comm& operator*();

  private:
    MPI_Comm communicator;
  };
  #endif

  /// This class provides utility functions for easy communcation with MPI.

  class MPI
  {
  public:

    /// Return proccess number
    static uint process_number();

    /// Return number of processes
    static uint num_processes();

    /// Determine whether we should broadcast (based on current parallel policy)
    static bool is_broadcaster();

    /// Determine whether we should receive (based on current parallel policy)
    static bool is_receiver();

    /// Set a barrier (synchronization point)
    static void barrier();

    // FIXME: Write documentation for this very fancy and versatile function!
    // FIXME: The mother of all MPI calls! It does everything anyone would ever
    //        need to do with MPI... :-)

    /// Distribute local arrays on all processors according to given partition
    template<typename T>
    static void distribute(const std::vector<T>& in_values,
                           const std::vector<uint>& destinations,
                           std::vector<T>& out_values,
                           std::vector<uint>& sources);

    /// Distribute local arrays on all processors according to given partition
    template<typename T>
    static void distribute(const std::vector<T>& in_values,
                           const std::vector<uint>& destinations,
                           std::vector<T>& out_values)
    {
      std::vector<uint> sources;
      distribute(in_values, destinations, out_values, sources);
    }

    // Broadcast value from broadcaster process to all processes
    template<typename T>
    static void broadcast(T& value, uint broadcaster=0)
    {
      #ifdef HAS_MPI
      MPICommunicator mpi_comm;
      boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);
      boost::mpi::broadcast(comm, value, broadcaster);
      #endif
    }

    /// Scatter in_values[i] to process i
    template<typename T>
    static void scatter(const std::vector<T>& in_values,
                        T& out_value, uint sending_process=0)
    {
      #ifdef HAS_MPI
      MPICommunicator mpi_comm;
      boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);
      boost::mpi::scatter(comm, in_values, out_value, sending_process);
      #else
      dolfin_assert(sending_process == 0);
      dolfin_assert(in_values.size() == 1);
      out_value = in_values[0];
      #endif
    }

    // Gather values on one process (wrapper for boost::mpi::gather)
    template<typename T>
    static void gather(const T& in_value, std::vector<T>& out_values,
                       uint receiving_process=0)
    {
      #ifdef HAS_MPI
      MPICommunicator mpi_comm;
      boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);
      boost::mpi::gather(comm, in_value, out_values, receiving_process);
      #else
      out_values.clear();
      out_values.push_back(in_value);
      #endif
    }

    // Gather values, one from each process (wrapper for boost::mpi::all_gather)
    template<typename T>
    static void all_gather(const T& in_value, std::vector<T>& out_values)
    {
      #ifdef HAS_MPI
      MPICommunicator mpi_comm;
      boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);
      boost::mpi::all_gather(comm, in_value, out_values);
      #else
      out_values.clear();
      out_values.push_back(in_value);
      #endif
    }

     // Return global max value
     template<typename T> static T max(const T& value)
     {
       #ifdef HAS_MPI
       return all_reduce(value, boost::mpi::maximum<T>());
       #else
       return value;
       #endif
     }

    // Return global min value
    template<typename T> static T min(const T& value)
    {
      #ifdef HAS_MPI
      return all_reduce(value, boost::mpi::minimum<T>());
      #else
      return value;
      #endif
    }

    // Sum values and return sum
    template<typename T> static T sum(const T& value)
    {
      #ifdef HAS_MPI
      return all_reduce(value, std::plus<T>());
      #else
      return value;
      #endif
    }

    // All reduce
    template<typename T, typename X> static T all_reduce(const T& value, X op)
    {
      #ifdef HAS_MPI
      MPICommunicator mpi_comm;
      boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);
      T out;
      boost::mpi::all_reduce(comm, value, out, op);
      return out;
      #else
      dolfin_error("MPI.h",
                   "call MPI::all_reduce",
                   "DOLFIN has been configured without MPI support");
      return T(0);
      #endif
    }

    /// Find global offset (index) (wrapper for MPI_(Ex)Scan with MPI_SUM as
    /// reduction op)
    static uint global_offset(uint range, bool exclusive);

    /// Send-receive data. Note that if the number of posted send-receives may
    /// differ between processes, a communicator MUST be supplied. (Duplicating
    /// the communicator requires participation of all processes.)
    template<typename T>
    static void send_recv(const T& send_value, uint dest,
                          T& recv_value, uint source)
    {
      #ifdef HAS_MPI
      MPICommunicator mpi_comm;
      send_recv(mpi_comm, send_value, dest, recv_value, source);
      #else
      dolfin_error("MPI.h",
                   "call MPI::send_recv",
                   "DOLFIN has been configured without MPI support");
      #endif
    }

    /// Send-receive data.
    template<typename T>
    static void send_recv(MPICommunicator& mpi_comm,
                          const T& send_value, uint dest,
                          T& recv_value, uint source)
    {
      #ifdef HAS_MPI
      boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);

      // Non-blocking send-receive
      boost::mpi::request reqs[2];
      reqs[0] = comm.isend(dest, 0, send_value);
      reqs[1] = comm.irecv(source, 0, recv_value);
      boost::mpi::wait_all(reqs, reqs + 2);
      #else
      dolfin_error("MPI.h",
                   "call MPI::send_recv",
                   "DOLFIN has been configured without MPI support");
      #endif
    }

    /// Return local range for local process, splitting [0, N - 1] into
    /// num_processes() portions of almost equal size
    static std::pair<uint, uint> local_range(uint N);

    /// Return local range for given process, splitting [0, N - 1] into
    /// num_processes() portions of almost equal size
    static std::pair<uint, uint> local_range(uint process, uint N);

    /// Return local range for given process, splitting [0, N - 1] into
    /// num_processes portions of almost equal size
    static std::pair<uint, uint> local_range(uint process, uint N,
                                             uint num_processes);

    /// Return which process owns index (inverse of local_range)
    static uint index_owner(uint index, uint N);

  private:

    #ifdef HAS_MPI
    // Return MPI data type
    template<typename T> static MPI_Datatype mpi_type()
    {
      dolfin_error("MPI.h",
                   "perform MPI operation",
                   "MPI data type unknown");
      return MPI_CHAR;
    }
    #endif

  };

  #ifdef HAS_MPI
  // Specialisations for MPI_Datatypes
  template<> inline MPI_Datatype MPI::mpi_type<double>() { return MPI_DOUBLE; }
  template<> inline MPI_Datatype MPI::mpi_type<int>() { return MPI_INT; }
  template<> inline MPI_Datatype MPI::mpi_type<long int>() { return MPI_LONG; }
  template<> inline MPI_Datatype MPI::mpi_type<unsigned int>() { return MPI_UNSIGNED; }
  template<> inline MPI_Datatype MPI::mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }
  #endif

  //#include "mpiutils.h"

  //---------------------------------------------------------------------------
  #ifdef HAS_MPI
  template<typename T>
  void MPI::distribute(const std::vector<T>& in_values,
                       const std::vector<uint>& destinations,
                       std::vector<T>& out_values,
                       std::vector<uint>& sources)
  {
    dolfin_assert(in_values.size() == destinations.size());

    // Get number of processes and process number
    const uint num_processes  = MPI::num_processes();
    const uint process_number = MPI::process_number();

    // Sort out data that should be sent to other processes
    std::vector<std::vector<T> > send_data(num_processes);
    for (uint i = 0; i < in_values.size(); i++)
    {
      // Get process number data should be sent to
      const uint p = destinations[i];
      dolfin_assert(p < send_data.size());

      // Append data to array for process p
      send_data[p].push_back(in_values[i]);
    }

    // Store local data (don't send) and clear partition vector and reuse for
    // storing sender of data
    out_values.clear();
    sources.clear();
    const std::vector<T>& local_values = send_data[process_number];
    for (uint i = 0; i < local_values.size(); i++)
    {
      out_values.push_back(local_values[i]);
      sources.push_back(process_number);
    }

    // Exchange data
    for (uint i = 1; i < send_data.size(); i++)
    {
      // We receive data from process p - i (i steps to the left)
      const int source = (process_number - i + num_processes) % num_processes;

      // We send data to process p + i (i steps to the right)
      const int dest = (process_number + i) % num_processes;

      // Send and receive data
      std::vector<T> recv_buffer;
      MPI::send_recv(send_data[dest], dest, recv_buffer, source);

      // Copy data from receive buffer
      out_values.insert(out_values.end(), recv_buffer.begin(), recv_buffer.end());
      sources.insert(sources.end(), recv_buffer.size(), source);
    }
  }
  #else
  template<typename T>
  void MPI::distribute(const std::vector<T>& in_values,
                       const std::vector<uint>& destinations,
                       std::vector<T>& out_values,
                       std::vector<uint>& sources)
  {
    dolfin_error("MPI.h",
                 "call MPI::distribute",
                 "DOLFIN has been configured without MPI support");
  }
  #endif
  //---------------------------------------------------------------------------

}

#endif
