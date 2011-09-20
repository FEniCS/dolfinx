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
// Modified by Garth N. Wells, 2008-2011.
// Modified by Ola Skavhaug, 2008-2009.
// Modified by Anders Logg, 2008-2009.
// Modified by Niclas Jansson, 2009.
//
// First added:  2007-11-30
// Last changed: 2011-06-30

#ifndef __MPI_DOLFIN_WRAPPER_H
#define __MPI_DOLFIN_WRAPPER_H

#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

// NOTE: It would be convenient to use Boost.MPI, but it is not yet well
//       supported by packaged versions of Boost. Boost.MPI code is therefore
//       commented out for now.

#ifdef HAS_MPI
//#include <boost/mpi.hpp>
#include <mpi.h>
#endif

namespace dolfin
{

  #ifdef HAS_MPI
  class MPICommunicator
  {

  public:

    /// Create communicator (copy of MPI_COMM_WORLD)
    MPICommunicator();

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
    static void distribute(std::vector<uint>& values,
                           std::vector<uint>& partition);

    /// Distribute local arrays on all processors according to given partition
    static void distribute(std::vector<double>& values,
                           std::vector<uint>& partition);

    // NOTE: This is commented out since Boost.MPI is not well supported on older platforms
    // // Broadcast value from broadcaster process to all processes
    // template<class T> static void broadcast(T& value, uint broadcaster=0)
    // {
    //   #ifdef HAS_MPI
    //   MPICommunicator mpi_comm;
    //   boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_duplicate);
    //   boost::mpi::broadcast(comm, value, broadcaster);
    //   #endif
    // }

    /// Broadcast value from broadcaster process to all processes
    template<class T> static void broadcast(T& value, uint broadcaster=0)
    {
      #ifdef HAS_MPI
      MPICommunicator comm;
      MPI_Bcast(&value, 1, mpi_type<T>(), broadcaster, *comm);
      #endif
    }

    /// Broadcast value from broadcaster process to all processes
    template<class T> static void broadcast(std::vector<T>& values, uint broadcaster=0)
    {
      #ifdef HAS_MPI
      // Communicate size
      uint size = values.size();
      broadcast(size, broadcaster);

      // Resize (will not affect broadcaster)
      values.resize(size);

      // Communicate
      MPICommunicator comm;
      MPI_Bcast(&values[0], size, mpi_type<T>(), broadcaster, *comm);
      #endif
    }

    // FIXME: Use common template function for uint and double scatter below

    /// Scatter values, one to each process
    static void scatter(std::vector<uint>& values, uint sending_process=0);

    /// Scatter values (wrapper for MPI_Scatterv)
    static void scatter(std::vector<std::vector<bool> >& values,
                        uint sending_process=0)
    { error("dolfin::MPI::scatter does not yet support bool."); }

    /// Scatter values (wrapper for MPI_Scatterv)
    static void scatter(std::vector<std::vector<uint> >& values,
                        uint sending_process=0);

    /// Scatter values (wrapper for MPI_Scatterv)
    static void scatter(std::vector<std::vector<int> >& values,
                        uint sending_process=0);

    /// Scatter values (wrapper for MPI_Scatterv)
    static void scatter(std::vector<std::vector<double> >& values,
                        uint sending_process=0);

    /// Gather values, one from each process (wrapper for MPI_Allgather)
    static std::vector<uint> gather(uint value);

    /// Gather values, one from each process (wrapper for MPI_Allgather)
    template<class T>
    static void gather(std::vector<T>& values)
    {
      #ifdef HAS_MPI
      assert(values.size() == num_processes());

      // Prepare arrays
      T send_value = values[process_number()];
      T* received_values = new T[values.size()];

      // Create communicator (copy of MPI_COMM_WORLD)
      MPICommunicator comm;

      // Call MPI
      MPI_Allgather(&send_value,     1, mpi_type<T>(),
                    received_values, 1, mpi_type<T>(), *comm);

      // Copy values
      for (uint i = 0; i < values.size(); i++)
        values[i] = received_values[i];

      // Cleanup
      delete [] received_values;
      #else
      error("MPI::gather() requires MPI.");
      #endif
    }

    // NOTE: This is commented out since Boost.MPI is not well supported
    //       on older platforms
    // // Gather values, one from each process (wrapper for boost::mpi::all_gather)
    // template<class T> static void gather_all(const T& in_value,
    //                                          std::vector<T>& out_values)
    // {
    //   #ifdef HAS_MPI
    //   MPICommunicator mpi_comm;
    //   boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_duplicate);
    //   boost::mpi::all_gather(comm, in_value, out_values);
    //   #else
    //   out_values.clear();
    //   #endif
    // }

    /// Return  maximum value
    template<class T> static T max(const T& value)
    {
      #ifdef HAS_MPI
      T _max(0);
      T _value = value;
      MPICommunicator comm;
      MPI_Allreduce(&_value, &_max, 1, mpi_type<T>(), MPI_MAX, *comm);
      return _max;
      #else
      return value;
      #endif
    }

    /// Return minimum value
    template<class T> static T min(const T& value)
    {
      #ifdef HAS_MPI
      T _min(0);
      T _value = value;
      MPICommunicator comm;
      MPI_Allreduce(&_value, &_min, 1, mpi_type<T>(), MPI_MIN, *comm);
      return _min;
      #else
      return value;
      #endif
    }

    /// Return sum across all processes
    template<class T> static T sum(const T& value)
    {
      #ifdef HAS_MPI
      T _sum(0);
      T _value = value;
      MPICommunicator comm;
      MPI_Allreduce(&_value, &_sum, 1, mpi_type<T>(), MPI_SUM, *comm);
      return _sum;
      #else
      return value;
      #endif
    }

    // NOTE: This is commented out since Boost.MPI is not well supported
    //       on older platforms
    // // Return global max value
    // template<class T> static T max(const T& value)
    // {
    //   #ifdef HAS_MPI
    //   return all_reduce(value, boost::mpi::maximum<T>());
    //   #else
    //   return value;
    //   #endif
    // }

    // // Return global min value
    // template<class T> static T min(const T& value)
    // {
    //   #ifdef HAS_MPI
    //   return all_reduce(value, boost::mpi::minimum<T>());
    //   #else
    //   return value;
    //   #endif
    // }

    // // Sum values and return sum
    // template<class T> static T sum(const T& value)
    // {
    //   #ifdef HAS_MPI
    //   return all_reduce(value, std::plus<T>());
    //   #else
    //   return value;
    //   #endif
    // }

    // // All reduce
    // template<class T, class X> static T all_reduce(const T& value, X op)
    // {
    //   #ifdef HAS_MPI
    //   MPICommunicator mpi_comm;
    //   boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_duplicate);
    //   T out;
    //   boost::mpi::all_reduce(comm, value, out, op);
    //   return out;
    //   #else
    //   error("MPI::all_reduce requires MPI to be configured.");
    //   return T(0);
    //   #endif
    // }
    //

    /// Find global offset (index) (wrapper for MPI_(Ex)Scan with MPI_SUM as
    /// reduction op)
    static uint global_offset(uint range, bool exclusive);

    /// Send-receive and return number of received values (wrapper for MPI_Sendrecv)
    static uint send_recv(uint* send_buffer, uint send_size, uint dest,
                          uint* recv_buffer, uint recv_size, uint source);

    /// Send-receive and return number of received values (wrapper for MPI_Sendrecv)
    static uint send_recv(double* send_buffer, uint send_size, uint dest,
                          double* recv_buffer, uint recv_size, uint source);

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
    template<class T> static MPI_Datatype mpi_type()
    {
      error("MPI data type unknown.");
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

}

#endif
