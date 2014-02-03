// Copyright (C) 2007-2013 Magnus Vikstrøm and Garth N. Wells
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
// Modified by Joachim B Haga 2012
//
// First added:  2007-11-30
// Last changed: 2013-01-11

#ifndef __MPI_DOLFIN_WRAPPER_H
#define __MPI_DOLFIN_WRAPPER_H

#include <numeric>
#include <utility>
#include <vector>

#ifdef HAS_MPI
#define MPICH_IGNORE_CXX_SEEK 1
#include <boost/serialization/utility.hpp>
#include <boost/mpi.hpp>
#include <mpi.h>
#endif

#include <dolfin/log/dolfin_log.h>

#ifndef HAS_MPI
typedef int MPI_Comm;
#define MPI_COMM_WORLD 2
#define MPI_COMM_SELF 1
#define MPI_COMM_NULL 0
#endif

namespace dolfin
{

  #ifdef HAS_MPI

  class MPIInfo
  {
  public:
    MPIInfo();
    ~MPIInfo();
    MPI_Info& operator*();
  private:

    MPI_Info info;

  };

  #endif

  class MPINonblocking
  {
    /// This class provides stateful (single communicator) non-blocking
    /// MPI functionality.

  public:

    /// Destroy instance (waits for outstanding requests)
    ~MPINonblocking()
    {
      #ifdef HAS_MPI
      wait_all();
      #endif
    }

    /// Non-blocking send and receive
    template<typename T>
      void send_recv(const MPI_Comm comm, const T& send_value,
                     unsigned int dest, T& recv_value, unsigned int source);

    /// Non-blocking send and receive with tag
    template<typename T>
      void send_recv(const MPI_Comm comm, const T& send_value,
                     unsigned int dest_tag, unsigned int dest,
                     T& recv_value, unsigned int source_tag,
                     unsigned int source);

    /// Wait for all requests to finish
    void wait_all();

  private:

    #ifdef HAS_MPI
    std::vector<boost::mpi::request> reqs;
    #endif

  };

  /// This class provides utility functions for easy communication
  /// with MPI and handles cases when DOLFIN is not configured with
  /// MPI.

  class MPI
  {
  public:

    /// Return process rank (uses MPI_COMM_WORLD)
    /// Warning: This function is deprecated. Use dolfin::MPI::rank
    static unsigned int process_number();

    /// Return number of processes for MPI_COMM_WORLD.
    /// Warning: This function is deprecated. Use dolfin::MPI::size.
    static unsigned int num_processes();

    /// Return process rank for the communicator
    static unsigned int rank(const MPI_Comm comm);

    /// Return size of the group (number of processes) associated with
    /// the communicator
    static unsigned int size(const MPI_Comm comm);

    /// Determine whether we should broadcast (based on current
    /// parallel policy)
    static bool is_broadcaster(const MPI_Comm comm);

    /// Determine whether we should receive (based on current parallel
    /// policy)
    static bool is_receiver(const MPI_Comm comm);

    /// Set a barrier (synchronization point)
    static void barrier(const MPI_Comm comm);

    /// Send in_values[p0] to process p0 and receive values from
    /// process p1 in out_values[p1]
    template<typename T>
      static void all_to_all(const MPI_Comm comm,
                             std::vector<std::vector<T> >& in_values,
                             std::vector<std::vector<T> >& out_values)
    {
      #ifdef HAS_MPI
      boost::mpi::communicator _comm(comm, boost::mpi::comm_attach);
      boost::mpi::all_to_all(_comm, in_values, out_values);
      #else
      dolfin_assert(in_values.size() == 1);
      out_values = in_values;
      #endif

    }

    /// Distribute local arrays on a group of processes (typically
    /// neighbours from GenericDofMap::neighbours()). It is important
    /// that each process' group includes exactly the processes that
    /// has it in their groups, otherwise it will deadlock.
    template<typename T, typename S>
      static void distribute(const MPI_Comm comm,
                             const std::set<S> group,
                             const std::map<S, T>& in_values_per_dest,
                             std::map<S, T>& out_values_per_src);


    /// Broadcast value from broadcaster process to all processes
    template<typename T>
      static void broadcast(const MPI_Comm comm, std::vector<T>& value,
                            unsigned int broadcaster=0)
    {
      #ifdef HAS_MPI
      // Broadcast cast size
      int bsize = value.size();
      MPI_Bcast(&bsize, 1, mpi_type<T>(), broadcaster, comm);

      // Broadcast
      value.resize(bsize);
      MPI_Bcast(const_cast<T*>(value.data()), bsize, mpi_type<T>(),
                broadcaster, comm);
      #endif
    }

    /// Broadcast value from broadcaster process to all processes
    template<typename T>
      static void broadcast(const MPI_Comm comm, T& value,
                            unsigned int broadcaster=0)
    {
      #ifdef HAS_MPI
      MPI_Bcast(&value, 1, mpi_type<T>(), broadcaster, comm);
      #endif
    }

    /// Scatter in_values[i] to process i
    template<typename T>
      static void scatter(const MPI_Comm comm,
                          const std::vector<std::vector<T> >& in_values,
                          std::vector<T>& out_value, unsigned int sending_process=0);

    /// Scatter in_values[i] to process i
    template<typename T>
      static void scatter(const MPI_Comm comm,
                          const std::vector<T>& in_values,
                          T& out_value, unsigned int sending_process=0);

    // NOTE: Part of removing Boost MPI transition
    /// Gather values on one process (wrapper for boost::mpi::gather)
    template<typename T>
      static void gather(const MPI_Comm comm, const std::vector<T>& in_values,
                         std::vector<T>& out_values,
                         unsigned int receiving_process=0)
    {
      #ifdef HAS_MPI
      const std::size_t comm_size = MPI::size(comm);

      // Get data size on each process
      std::vector<int> pcounts(comm_size);
      const int local_size = in_values.size();
      MPI_Gather(const_cast<int*>(&local_size), 1, mpi_type<int>(),
                 pcounts.data(), 1, mpi_type<int>(),
                 receiving_process, comm);

      // Build offsets
      std::vector<int> offsets(comm_size + 1, 0);
      for (std::size_t i = 1; i <= comm_size; ++i)
        offsets[i] = offsets[i - 1] + pcounts[i - 1];

      const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
      out_values.resize(n);
      MPI_Gatherv(const_cast<T*>(in_values.data()), in_values.size(), mpi_type<T>(),
                  out_values.data(), pcounts.data(), offsets.data(),
                  mpi_type<T>(), receiving_process, comm);
      #else
      out_values = in_values;
      #endif
    }

    /// Gather strings on one process
    static void gather(const MPI_Comm comm, const std::string& in_values,
                       std::vector<std::string>& out_values,
                       unsigned int receiving_process=0)
    {
      #ifdef HAS_MPI
      const std::size_t comm_size = MPI::size(comm);

      // Get data size on each process
      std::vector<int> pcounts(comm_size);
      int local_size = in_values.size();
      MPI_Gather(&local_size, 1, MPI_INT,
                 pcounts.data(), 1,MPI_INT,
                 receiving_process, comm);

      // Build offsets
      std::vector<int> offsets(comm_size + 1, 0);
      for (std::size_t i = 1; i <= comm_size; ++i)
        offsets[i] = offsets[i - 1] + pcounts[i - 1];

      // Gather
      const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
      std::vector<char> _out(n);
      MPI_Gatherv(const_cast<char*>(in_values.data()), in_values.size(), MPI_CHAR,
                  _out.data(), pcounts.data(), offsets.data(),
                  MPI_CHAR, receiving_process, comm);

      // Rebuild
      out_values.resize(comm_size);
      for (std::size_t p = 0; p < comm_size; ++p)
        out_values[p] = std::string(_out.begin() + offsets[p], _out.begin() + + offsets[p + 1]);

      #else
      out_values.clear();
      out_values.push_back(in_values);
      #endif
    }

    /// Gather values from all proceses. Same data count from each
    /// process (wrapper for MPI_Allgather)
    template<typename T>
      static void all_gather(const MPI_Comm comm, const std::vector<T>& in_values,
                             std::vector<T>& out_values)
    {
      #ifdef HAS_MPI
      out_values.resize(in_values.size()*MPI::size(comm));
      MPI_Allgather(const_cast<T*>(in_values.data()), in_values.size(), mpi_type<T>(),
                    out_values.data(), in_values.size(), mpi_type<T>(),
                    comm);
      #else
      out_values = in_values;
      #endif
    }

    /// Gather values from each process (variable count per process)
    template<typename T>
      static void all_gather(const MPI_Comm comm, const std::vector<T>& in_values,
                              std::vector<std::vector<T> >& out_values)
    {
      #ifdef HAS_MPI
      const std::size_t comm_size = MPI::size(comm);

      // Get data size on each process
      std::vector<int> pcounts;
      const int local_size = in_values.size();
      MPI::all_gather(comm, local_size, pcounts);
      dolfin_assert(pcounts.size() == comm_size);

      // Build offsets
      std::vector<int> offsets(comm_size + 1, 0);
      for (std::size_t i = 1; i <= comm_size; ++i)
        offsets[i] = offsets[i - 1] + pcounts[i - 1];

      // Gather data
      const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
      std::vector<T> recvbuf(n);
      MPI_Allgatherv(const_cast<T*>(in_values.data()), in_values.size(), mpi_type<T>(),
                     recvbuf.data(), pcounts.data(), offsets.data(),
                     mpi_type<T>(), comm);

      // Repack data
      out_values.resize(comm_size);
      for (std::size_t p = 0; p < comm_size; ++p)
      {
        out_values[p].resize(pcounts[p]);
        for (std::size_t i = 0; i < pcounts[p]; ++i)
          out_values[p][i] = recvbuf[offsets[p] + i];
      }
      #else
      out_values.clear();
      out_values.push_back(in_value);
      #endif
    }

    /// Gather values, one primitive from each process (MPI_Allgather)
    template<typename T>
      static void all_gather(const MPI_Comm comm, const T& in_value,
                             std::vector<T>& out_values)
    {
      #ifdef HAS_MPI
      out_values.resize(MPI::size(comm));
      MPI_Allgather(const_cast<T*>(&in_value), 1, mpi_type<T>(),
                    out_values.data(), 1, mpi_type<T>(), comm);
      #else
      out_values.clear();
      out_values.push_back(in_value);
      #endif
    }

    template<typename T> static T max(const MPI_Comm comm, const T& value)
    {
      #ifdef HAS_MPI
      return all_reduce(comm, value, MPI_MAX);
      #else
      return value;
      #endif
    }

    /// Return global min value
    template<typename T> static T min(const MPI_Comm comm, const T& value)
    {
      #ifdef HAS_MPI
      return all_reduce(comm, value, MPI_MIN);
      #else
      return value;
      #endif
    }

    /// Sum values and return sum
    template<typename T> static T sum(const MPI_Comm comm, const T& value)
    {
      #ifdef HAS_MPI
      return all_reduce(comm, value, MPI_SUM);
      #else
      return value;
      #endif
    }

    /// All reduce
    template<typename T, typename X> static
      T all_reduce(const MPI_Comm comm, const T& value, X op)
    {
      #ifdef HAS_MPI
      T out;
      MPI_Allreduce(const_cast<T*>(&value), &out, 1, mpi_type<T>(), op, comm);
      return out;
      #else
      dolfin_error("MPI.h",
                   "call MPI::all_reduce",
                   "DOLFIN has been configured without MPI support");
      return T(0);
      #endif
    }

    /// Find global offset (index) (wrapper for MPI_(Ex)Scan with
    /// MPI_SUM as reduction op)
    static std::size_t global_offset(const MPI_Comm comm,
                                     std::size_t range, bool exclusive);

    /// Send-receive data. Note that if the number of posted
    /// send-receives may differ between processes, another interface
    /// (such as MPINonblocking::send_recv) must be used since
    /// duplicating the communicator requires participation from all
    /// processes.
    template<typename T>
      static void send_recv(const MPI_Comm comm,
                            const T& send_value, unsigned int dest,
                            T& recv_value, unsigned int source)
    {
      #ifdef HAS_MPI
      MPINonblocking mpi;
      mpi.send_recv(comm, send_value, dest, recv_value, source);
      #else
      dolfin_error("MPI.h",
                   "call MPI::send_recv",
                   "DOLFIN has been configured without MPI support");
      #endif
    }

    /// Return local range for local process, splitting [0, N - 1] into
    /// size() portions of almost equal size
    static std::pair<std::size_t, std::size_t>
      local_range(const MPI_Comm comm, std::size_t N);

    /// Return local range for given process, splitting [0, N - 1] into
    /// size() portions of almost equal size
    static std::pair<std::size_t, std::size_t>
      local_range(const MPI_Comm comm, unsigned int process,
                  std::size_t N);

    /// Return local range for given process, splitting [0, N - 1] into
    /// size() portions of almost equal size
    static std::pair<std::size_t, std::size_t>
      compute_local_range(unsigned int process, std::size_t N,
                          unsigned int size);

    /// Return which process owns index (inverse of local_range)
    static unsigned int index_owner(const MPI_Comm comm,
                                    std::size_t index, std::size_t N);

  private:

    #ifndef HAS_MPI
    static void error_no_mpi(const char *where)
    {
      dolfin_error("MPI.h", where, "DOLFIN has been configured without MPI support");
    }
    #endif

    #ifdef HAS_MPI
    // Return MPI data type
    template<typename T> static MPI_Datatype mpi_type();
    /*
    template<typename T> static MPI_Datatype mpi_type()
    {
      dolfin_error("MPI.h",
                   "perform MPI operation",
                   "MPI data type unknown");
      return MPI_CHAR;
    }
    */
    #endif

  };

  #ifdef HAS_MPI
  // Specialisations for MPI_Datatypes
  template<> inline MPI_Datatype MPI::mpi_type<float>() { return MPI_FLOAT; }
  template<> inline MPI_Datatype MPI::mpi_type<double>() { return MPI_DOUBLE; }
  template<> inline MPI_Datatype MPI::mpi_type<short int>() { return MPI_SHORT; }
  template<> inline MPI_Datatype MPI::mpi_type<int>() { return MPI_INT; }
  template<> inline MPI_Datatype MPI::mpi_type<long int>() { return MPI_LONG; }
  template<> inline MPI_Datatype MPI::mpi_type<unsigned int>() { return MPI_UNSIGNED; }
  template<> inline MPI_Datatype MPI::mpi_type<unsigned long int>() { return MPI_UNSIGNED_LONG; }
  #endif

  //---------------------------------------------------------------------------
  template<typename T, typename S>
    void dolfin::MPI::distribute(const MPI_Comm comm,
                                 const std::set<S> processes_group,
                                 const std::map<S, T>& in_values_per_dest,
                                 std::map<S, T>& out_values_per_src)
  {
    #ifdef HAS_MPI
    typedef typename std::map<S, T>::const_iterator map_const_iterator;
    typedef typename std::map<S, T>::iterator map_iterator;
    dolfin::MPINonblocking mpi;
    const T no_data;

    // Send and receive values to all processes in groups
    // (non-blocking). If a given process is not found in
    // in_values_per_dest, send empty data.
    out_values_per_src.clear();
    typename std::set<S>::const_iterator dest;
    for (dest = processes_group.begin(); dest != processes_group.end(); ++dest)
    {
      map_const_iterator values = in_values_per_dest.find(*dest);
      if (values != in_values_per_dest.end())
      {
        mpi.send_recv(comm, values->second, *dest,
                      out_values_per_src[*dest], *dest);
      }
      else
      {
        mpi.send_recv(comm, no_data, *dest, out_values_per_src[*dest],
                      *dest);
      }
    }

    // Wait for all MPI calls before modifying out_values_per_src
    mpi.wait_all();

    // Remove received no_data entries.
    map_iterator it = out_values_per_src.begin();
    while (it != out_values_per_src.end())
    {
      map_iterator tmp = it++;
      if (tmp->second.empty())
        out_values_per_src.erase(tmp);
    }
    #else
    error_no_mpi("call MPI::distribute");
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPINonblocking::send_recv(const MPI_Comm comm,
                                           const T& send_value,
                                           unsigned int dest,
                                           T& recv_value, unsigned int source)
  {
    MPINonblocking::send_recv(comm, send_value, 0, dest, recv_value, 0,
                              source);
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPINonblocking::send_recv(const MPI_Comm comm,
                                           const T& send_value,
                                           unsigned int dest_tag,
                                           unsigned int dest,
                                           T& recv_value,
                                           unsigned int source_tag,
                                           unsigned int source)
  {
    #ifdef HAS_MPI
    boost::mpi::communicator _comm(comm, boost::mpi::comm_attach);
    reqs.push_back(_comm.isend(dest, dest_tag, send_value));
    reqs.push_back(_comm.irecv(source, source_tag, recv_value));
    #else
    dolfin_error("MPI.h",
                  "call MPINonblocking::send_recv",
                  "DOLFIN has been configured without MPI support");
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::scatter(const MPI_Comm comm,
                              const std::vector<std::vector<T> >& in_values,
                              std::vector<T>& out_value,
                              unsigned int sending_process)
  {
    #ifdef HAS_MPI

    // Scatter number of values to each process
    const std::size_t comm_size = MPI::size(comm);
    std::vector<int> all_num_values;
    if (MPI::rank(comm) == sending_process)
    {
      dolfin_assert(in_values.size() == comm_size);
      all_num_values.resize(comm_size);
      for (std::size_t i = 0; i < comm_size; ++i)
        all_num_values[i] = in_values[i].size();
    }
    int my_num_values = 0;
    scatter(comm, all_num_values, my_num_values, sending_process);

    // Prepare send buffer and offsets
    std::vector<T> sendbuf;
    std::vector<int> offsets;
    if (MPI::rank(comm) == sending_process)
    {
      // Build offsets
      offsets.resize(comm_size + 1, 0);
      for (std::size_t i = 1; i <= comm_size; ++i)
        offsets[i] = offsets[i - 1] + all_num_values[i - 1];

      // Allocate send buffer and fill
      const std::size_t n = std::accumulate(all_num_values.begin(),
                                            all_num_values.end(), 0);
      sendbuf.resize(n);
      for (std::size_t p = 0; p < in_values.size(); ++p)
      {
        std::copy(in_values[p].begin(), in_values[p].end(),
                  sendbuf.begin() + offsets[p]);
        }
    }

    // Scatter
    out_value.resize(my_num_values);
    MPI_Scatterv(const_cast<T*>(sendbuf.data()), all_num_values.data(),
                 offsets.data(), mpi_type<T>(),
                 out_value.data(), my_num_values,
                 mpi_type<T>(), sending_process, comm);
    #else
    dolfin_assert(sending_process == 0);
    dolfin_assert(in_values.size() == 1);
    out_value = in_values[0];
    #endif
  }
  //---------------------------------------------------------------------------
  template<> inline
    void dolfin::MPI::scatter(const MPI_Comm comm,
                              const std::vector<std::vector<bool> >& in_values,
                              std::vector<bool>& out_value,
                              unsigned int sending_process)
  {
    #ifdef HAS_MPI
    // Copy data
    std::vector<std::vector<short int> > in(in_values.size());
    for (std::size_t i = 0; i < in_values.size(); ++i)
      in[i] = std::vector<short int>(in_values[i].begin(), in_values[i].end());

    std::vector<short int> out;
    scatter(comm, in, out, sending_process);

    out_value.resize(out.size());
    std::copy(out.begin(), out.end(), out_value.begin());
    #else
    dolfin_assert(sending_process == 0);
    dolfin_assert(in_values.size() == 1);
    out_value = in_values[0];
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::scatter(const MPI_Comm comm,
                              const std::vector<T>& in_values,
                              T& out_value, unsigned int sending_process)
  {
    #ifdef HAS_MPI
    boost::mpi::communicator _comm(comm, boost::mpi::comm_attach);
    boost::mpi::scatter(_comm, in_values, out_value, sending_process);
    return ;
    if (MPI::rank(comm) == sending_process)
      dolfin_assert(in_values.size() == MPI::size(comm));

    MPI_Scatter(const_cast<T*>(in_values.data()), 1, mpi_type<T>(),
                 &out_value, 1, mpi_type<T>(), sending_process, comm);
    #else
    dolfin_assert(sending_process == 0);
    dolfin_assert(in_values.size() == 1);
    out_value = in_values[0];
    #endif
  }
  //---------------------------------------------------------------------------

}

#endif
