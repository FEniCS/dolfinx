// Copyright (C) 2007-2014 Magnus Vikstrøm and Garth N. Wells
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
// Modified by Martin Sandve Alnes 2014

#ifndef __MPI_DOLFIN_WRAPPER_H
#define __MPI_DOLFIN_WRAPPER_H

#include <cstdint>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef HAS_MPI
#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>
#endif

#include <dolfin/log/log.h>
#include <dolfin/log/Table.h>

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

  /// This class provides utility functions for easy communication
  /// with MPI and handles cases when DOLFIN is not configured with
  /// MPI.

  class MPI
  {
  public:

    /// Return process rank for the communicator
    static unsigned int rank(MPI_Comm comm);

    /// Return size of the group (number of processes) associated with
    /// the communicator
    static unsigned int size(MPI_Comm comm);

    /// Determine whether we should broadcast (based on current
    /// parallel policy)
    static bool is_broadcaster(MPI_Comm comm);

    /// Determine whether we should receive (based on current parallel
    /// policy)
    static bool is_receiver(MPI_Comm comm);

    /// Set a barrier (synchronization point)
    static void barrier(MPI_Comm comm);

    /// Send in_values[p0] to process p0 and receive values from
    /// process p1 in out_values[p1]
    template<typename T>
      static void all_to_all(MPI_Comm comm,
                             std::vector<std::vector<T> >& in_values,
                             std::vector<std::vector<T> >& out_values);

    /// Broadcast vector of value from broadcaster to all processes
    template<typename T>
      static void broadcast(MPI_Comm comm, std::vector<T>& value,
                            unsigned int broadcaster=0);

    /// Broadcast single primitive from broadcaster to all processes
    template<typename T>
      static void broadcast(MPI_Comm comm, T& value,
                            unsigned int broadcaster=0);

    /// Scatter vector in_values[i] to process i
    template<typename T>
      static void scatter(MPI_Comm comm,
                          const std::vector<std::vector<T> >& in_values,
                          std::vector<T>& out_value,
                          unsigned int sending_process=0);

    /// Scatter primitive in_values[i] to process i
    template<typename T>
      static void scatter(MPI_Comm comm,
                          const std::vector<T>& in_values,
                          T& out_value, unsigned int sending_process=0);

    /// Gather values on one process
    template<typename T>
      static void gather(MPI_Comm comm, const std::vector<T>& in_values,
                         std::vector<T>& out_values,
                         unsigned int receiving_process=0);

    /// Gather strings on one process
    static void gather(MPI_Comm comm, const std::string& in_values,
                       std::vector<std::string>& out_values,
                       unsigned int receiving_process=0);

    /// Gather values from all processes. Same data count from each
    /// process (wrapper for MPI_Allgather)
    template<typename T>
      static void all_gather(MPI_Comm comm,
                             const std::vector<T>& in_values,
                             std::vector<T>& out_values);

    /// Gather values from each process (variable count per process)
    template<typename T>
      static void all_gather(MPI_Comm comm,
                             const std::vector<T>& in_values,
                             std::vector<std::vector<T> >& out_values);

    /// Gather values, one primitive from each process (MPI_Allgather)
    template<typename T>
      static void all_gather(MPI_Comm comm, const T in_value,
                             std::vector<T>& out_values);

    /// Return global max value
    template<typename T> static T max(MPI_Comm comm, const T& value);

    /// Return global min value
    template<typename T> static T min(MPI_Comm comm, const T& value);

    /// Sum values and return sum
    template<typename T> static T sum(MPI_Comm comm, const T& value);

    /// Return average across comm; implemented only for T == Table
    template<typename T> static T avg(MPI_Comm comm, const T& value);

    /// All reduce
    template<typename T, typename X> static
      T all_reduce(MPI_Comm comm, const T& value, X op);

    /// Find global offset (index) (wrapper for MPI_(Ex)Scan with
    /// MPI_SUM as reduction op)
    static std::size_t global_offset(MPI_Comm comm,
                                     std::size_t range, bool exclusive);

    /// Send-receive data between processes (blocking)
    template<typename T>
      static void send_recv(MPI_Comm comm,
                            const std::vector<T>& send_value,
                            unsigned int dest, int send_tag,
                            std::vector<T>& recv_value,
                            unsigned int source, int recv_tag);

    /// Send-receive data between processes
    template<typename T>
      static void send_recv(MPI_Comm comm,
                            const std::vector<T>& send_value, unsigned int dest,
                            std::vector<T>& recv_value, unsigned int source);

    /// Return local range for local process, splitting [0, N - 1] into
    /// size() portions of almost equal size
    static std::pair<std::int64_t, std::int64_t>
      local_range(MPI_Comm comm, std::int64_t N);

    /// Return local range for given process, splitting [0, N - 1] into
    /// size() portions of almost equal size
    static std::pair<std::int64_t, std::int64_t>
      local_range(MPI_Comm comm, int process, std::int64_t N);

    /// Return local range for given process, splitting [0, N - 1] into
    /// size() portions of almost equal size
    static std::pair<std::int64_t, std::int64_t>
      compute_local_range(int process, std::int64_t N, int size);

    /// Return which process owns index (inverse of local_range)
    static unsigned int index_owner(MPI_Comm comm,
                                    std::size_t index, std::size_t N);

    #ifdef HAS_MPI
    /// Return average reduction operation; recognized by
    /// all_reduce(MPI_Comm, Table&, MPI_Op)
    static MPI_Op MPI_AVG();
    #endif

  private:

    #ifndef HAS_MPI
    static void error_no_mpi(const char *where)
    {
      dolfin_error("MPI.h",
                   where,
                   "DOLFIN has been configured without MPI support");
    }
    #endif

    #ifdef HAS_MPI
    // Return MPI data type
    template<typename T>
      struct dependent_false : std::false_type {};
    template<typename T> static MPI_Datatype mpi_type()
    {
      static_assert(dependent_false<T>::value, "Unknown MPI type");
      dolfin_error("MPI.h",
                   "perform MPI operation",
                   "MPI data type unknown");
      return MPI_CHAR;
    }
    #endif

    #ifdef HAS_MPI
    // Maps some MPI_Op values to string
    static std::map<MPI_Op, std::string> operation_map;
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
  template<> inline MPI_Datatype MPI::mpi_type<long long>() { return MPI_LONG_LONG; }
  #endif
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::broadcast(MPI_Comm comm, std::vector<T>& value,
                                unsigned int broadcaster)
  {
    #ifdef HAS_MPI
    // Broadcast cast size
    std::size_t bsize = value.size();
    MPI_Bcast(&bsize, 1, mpi_type<std::size_t>(), broadcaster, comm);

    // Broadcast
    value.resize(bsize);
    MPI_Bcast(const_cast<T*>(value.data()), bsize, mpi_type<T>(),
              broadcaster, comm);
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::broadcast(MPI_Comm comm, T& value,
                                unsigned int broadcaster)
  {
    #ifdef HAS_MPI
    MPI_Bcast(&value, 1, mpi_type<T>(), broadcaster, comm);
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::all_to_all(MPI_Comm comm,
                                 std::vector<std::vector<T> >& in_values,
                                 std::vector<std::vector<T> >& out_values)
  {
    #ifdef HAS_MPI
    const std::size_t comm_size = MPI::size(comm);

    // Data size per destination
    dolfin_assert(in_values.size() == comm_size);
    std::vector<int> data_size_send(comm_size);
    std::vector<int> data_offset_send(comm_size + 1, 0);
    for (std::size_t p = 0; p < comm_size; ++p)
    {
      data_size_send[p] = in_values[p].size();
      data_offset_send[p + 1] = data_offset_send[p] + data_size_send[p];
    }

    // Get received data sizes
    std::vector<int> data_size_recv(comm_size);
    MPI_Alltoall(data_size_send.data(), 1, mpi_type<int>(),
                 data_size_recv.data(), 1, mpi_type<int>(),
                 comm);

    // Pack data and build receive offset
    std::vector<int> data_offset_recv(comm_size + 1 ,0);
    std::vector<T> data_send(data_offset_send[comm_size]);
    for (std::size_t p = 0; p < comm_size; ++p)
    {
      data_offset_recv[p + 1] = data_offset_recv[p] + data_size_recv[p];
      std::copy(in_values[p].begin(), in_values[p].end(),
                data_send.begin() + data_offset_send[p]);
    }

    // Send/receive data
    std::vector<T> data_recv(data_offset_recv[comm_size]);
    MPI_Alltoallv(data_send.data(), data_size_send.data(),
                  data_offset_send.data(), mpi_type<T>(),
                  data_recv.data(), data_size_recv.data(),
                  data_offset_recv.data(), mpi_type<T>(), comm);

      // Repack data
    out_values.resize(comm_size);
    for (std::size_t p = 0; p < comm_size; ++p)
    {
      out_values[p].resize(data_size_recv[p]);
      std::copy(data_recv.begin() + data_offset_recv[p],
                data_recv.begin() + data_offset_recv[p + 1],
                out_values[p].begin());
    }
    #else
    dolfin_assert(in_values.size() == 1);
    out_values = in_values;
    #endif
  }
  //---------------------------------------------------------------------------
  template<> inline
    void dolfin::MPI::all_to_all(MPI_Comm comm,
                                 std::vector<std::vector<bool> >& in_values,
                                 std::vector<std::vector<bool> >& out_values)
  {
    #ifdef HAS_MPI
    // Copy to short int
    std::vector<std::vector<short int> > send(in_values.size());
    for (std::size_t i = 0; i < in_values.size(); ++i)
      send[i].assign(in_values[i].begin(), in_values[i].end());

    // Communicate data
    std::vector<std::vector<short int> > recv;
    all_to_all(comm, send, recv);

    // Copy back to bool
    out_values.resize(recv.size());
    for (std::size_t i = 0; i < recv.size(); ++i)
      out_values[i].assign(recv[i].begin(), recv[i].end());
    #else
    dolfin_assert(in_values.size() == 1);
    out_values = in_values;
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::scatter(MPI_Comm comm,
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
    void dolfin::MPI::scatter(MPI_Comm comm,
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
    void dolfin::MPI::scatter(MPI_Comm comm,
                              const std::vector<T>& in_values,
                              T& out_value, unsigned int sending_process)
  {
    #ifdef HAS_MPI
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
  template<typename T>
  void dolfin::MPI::gather(MPI_Comm comm,
                           const std::vector<T>& in_values,
                           std::vector<T>& out_values,
                           unsigned int receiving_process)
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
    MPI_Gatherv(const_cast<T*>(in_values.data()), in_values.size(),
                mpi_type<T>(),
                out_values.data(), pcounts.data(), offsets.data(),
                mpi_type<T>(), receiving_process, comm);
    #else
    out_values = in_values;
    #endif
  }
  //---------------------------------------------------------------------------
  inline void dolfin::MPI::gather(MPI_Comm comm,
                                  const std::string& in_values,
                                  std::vector<std::string>& out_values,
                                  unsigned int receiving_process)
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
    MPI_Gatherv(const_cast<char*>(in_values.data()), in_values.size(),
                MPI_CHAR,
                _out.data(), pcounts.data(), offsets.data(),
                MPI_CHAR, receiving_process, comm);

    // Rebuild
    out_values.resize(comm_size);
    for (std::size_t p = 0; p < comm_size; ++p)
    {
      out_values[p] = std::string(_out.begin() + offsets[p],
                                  _out.begin() + offsets[p + 1]);
    }
    #else
    out_values.clear();
    out_values.push_back(in_values);
    #endif
  }
  //-------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::all_gather(MPI_Comm comm,
                                 const std::vector<T>& in_values,
                                 std::vector<T>& out_values)
  {
    #ifdef HAS_MPI
    out_values.resize(in_values.size()*MPI::size(comm));
    MPI_Allgather(const_cast<T*>(in_values.data()), in_values.size(),
                  mpi_type<T>(),
                  out_values.data(), in_values.size(), mpi_type<T>(),
                  comm);
    #else
    out_values = in_values;
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::all_gather(MPI_Comm comm,
                                 const std::vector<T>& in_values,
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
    MPI_Allgatherv(const_cast<T*>(in_values.data()), in_values.size(),
                   mpi_type<T>(),
                   recvbuf.data(), pcounts.data(), offsets.data(),
                   mpi_type<T>(), comm);

    // Repack data
    out_values.resize(comm_size);
    for (std::size_t p = 0; p < comm_size; ++p)
    {
      out_values[p].resize(pcounts[p]);
      for (int i = 0; i < pcounts[p]; ++i)
        out_values[p][i] = recvbuf[offsets[p] + i];
    }
    #else
    out_values.clear();
    out_values.push_back(in_values);
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::all_gather(MPI_Comm comm, const T in_value,
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
  //---------------------------------------------------------------------------
  template<typename T, typename X>
    T dolfin::MPI::all_reduce(MPI_Comm comm, const T& value, X op)
  {
    #ifdef HAS_MPI
    T out;
    MPI_Allreduce(const_cast<T*>(&value), &out, 1, mpi_type<T>(), op, comm);
    return out;
    #else
    return value;
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T> T dolfin::MPI::max(MPI_Comm comm, const T& value)
  {
    #ifdef HAS_MPI
    // Enforce cast to MPI_Op; this is needed because template dispatch may
    // not recognize this is possible, e.g. C-enum to unsigned int in SGI MPT
    MPI_Op op = static_cast<MPI_Op>(MPI_MAX);
    return all_reduce(comm, value, op);
    #else
    return value;
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T> T dolfin::MPI::min(MPI_Comm comm, const T& value)
  {
    #ifdef HAS_MPI
    // Enforce cast to MPI_Op; this is needed because template dispatch may
    // not recognize this is possible, e.g. C-enum to unsigned int in SGI MPT
    MPI_Op op = static_cast<MPI_Op>(MPI_MIN);
    return all_reduce(comm, value, op);
    #else
    return value;
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T> T dolfin::MPI::sum(MPI_Comm comm, const T& value)
  {
    #ifdef HAS_MPI
    // Enforce cast to MPI_Op; this is needed because template dispatch may
    // not recognize this is possible, e.g. C-enum to unsigned int in SGI MPT
    MPI_Op op = static_cast<MPI_Op>(MPI_SUM);
    return all_reduce(comm, value, op);
    #else
    return value;
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T> T dolfin::MPI::avg(MPI_Comm comm, const T& value)
  {
    #ifdef HAS_MPI
    dolfin_error("MPI.h",
                 "perform average reduction",
                 "Not implemented for this type");
    #else
    return value;
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::send_recv(MPI_Comm comm,
                                const std::vector<T>& send_value,
                                unsigned int dest, int send_tag,
                                std::vector<T>& recv_value,
                                unsigned int source, int recv_tag)
  {
    #ifdef HAS_MPI
    std::size_t send_size = send_value.size();
    std::size_t recv_size = 0;
    MPI_Status mpi_status;
    MPI_Sendrecv(&send_size, 1, mpi_type<std::size_t>(), dest, send_tag,
                 &recv_size, 1, mpi_type<std::size_t>(), source, recv_tag,
                 comm, &mpi_status);

    recv_value.resize(recv_size);
    MPI_Sendrecv(const_cast<T*>(send_value.data()), send_value.size(),
                 mpi_type<T>(), dest, send_tag,
                 recv_value.data(), recv_size, mpi_type<T>(),
                 source, recv_tag, comm, &mpi_status);
    #else
      dolfin_error("MPI.h",
      "call MPI::send_recv",
      "DOLFIN has been configured without MPI support");
    #endif
  }
  //---------------------------------------------------------------------------
  template<typename T>
    void dolfin::MPI::send_recv(MPI_Comm comm,
                                const std::vector<T>& send_value,
                                unsigned int dest,
                                std::vector<T>& recv_value,
                                unsigned int source)
  {
    MPI::send_recv(comm, send_value, dest, 0, recv_value, source, 0);
  }
  //---------------------------------------------------------------------------
  // Specialization for dolfin::log::Table class
  // NOTE: This function is not trully "all_reduce", it reduces to rank 0
  //       and returns zero Table on other ranks.
  #ifdef HAS_MPI
  template<>
    Table dolfin::MPI::all_reduce(MPI_Comm, const Table&, MPI_Op);
  #endif
  //---------------------------------------------------------------------------
  // Specialization for dolfin::log::Table class
  #ifdef HAS_MPI
  template<>
    Table dolfin::MPI::avg(MPI_Comm, const Table&);
  #endif
  //---------------------------------------------------------------------------
}

#endif
