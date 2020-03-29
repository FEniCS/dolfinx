// Copyright (C) 2007-2014 Magnus Vikstrøm and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>

namespace dolfinx
{

/// This class provides utility functions for easy communication with
/// MPI and handles cases when DOLFINX is not configured with MPI.
class MPI
{
public:
  /// A duplicate MPI communicator and manage lifetime of the
  /// communicator
  class Comm
  {
  public:
    /// Duplicate communicator and wrap duplicate
    Comm(MPI_Comm comm);

    /// Copy constructor
    Comm(const Comm& comm);

    /// Move constructor
    Comm(Comm&& comm) noexcept;

    // Disable copy assignment operator
    Comm& operator=(const Comm& comm) = delete;

    /// Move assignment operator
    Comm& operator=(Comm&& comm);

    /// Destructor (frees wrapped communicator)
    ~Comm();

    /// Return the underlying MPI_Comm object
    MPI_Comm comm() const;

  private:
    // MPI communicator
    MPI_Comm _comm;
  };

  /// Create a new comm with a subset of processes
  static MPI_Comm SubsetComm(MPI_Comm comm, int num_processes);

  /// Return process rank for the communicator
  static std::uint32_t rank(MPI_Comm comm);

  /// Return size of the group (number of processes) associated with the
  /// communicator
  static std::uint32_t size(MPI_Comm comm);

  /// Set a barrier (synchronization point)
  static void barrier(MPI_Comm comm);

private:
  // Implementation of all_to_all, common for both cases, whether
  // returning a flat array, or in separate vectors by sending process.
  template <typename T>
  static void all_to_all_common(MPI_Comm comm,
                                const std::vector<std::vector<T>>& in_values,
                                std::vector<T>& out_values,
                                std::vector<std::int32_t>& offsets);

public:
  /// Send in_values[p0] to process p0 and receive values from process
  /// p1 in out_values[p1]
  template <typename T>
  static void all_to_all(MPI_Comm comm,
                         const std::vector<std::vector<T>>& in_values,
                         std::vector<std::vector<T>>& out_values);

  /// Send in_values[p0] to process p0 and receive values from all
  /// processes in out_values
  template <typename T>
  static void all_to_all(MPI_Comm comm,
                         const std::vector<std::vector<T>>& in_values,
                         std::vector<T>& out_values);

  /// Neighbourhood all-to-all. Send data to neighbours using offsets into
  /// contiguous data array. Offset array should contain (num_neighbours + 1)
  /// entries, starting from zero.
  template <typename T>
  static void neighbor_all_to_all(MPI_Comm neighbor_comm,
                                  const std::vector<int>& send_offsets,
                                  const std::vector<T>& send_data,
                                  std::vector<int>& recv_offsets,
                                  std::vector<T>& recv_data);

  /// Return list of neighbours for a neighbourhood comm
  /// @param neighbor_comm
  static std::vector<int> neighbors(MPI_Comm neighbor_comm);

  /// Broadcast vector of value from broadcaster to all processes
  template <typename T>
  static void broadcast(MPI_Comm comm, std::vector<T>& value,
                        std::uint32_t broadcaster = 0);

  /// Broadcast single primitive from broadcaster to all processes
  template <typename T>
  static void broadcast(MPI_Comm comm, T& value, std::uint32_t broadcaster = 0);

  /// Scatter vector in_values[i] to process i
  template <typename T>
  static void
  scatter(MPI_Comm comm, const std::vector<std::vector<T>>& in_values,
          std::vector<T>& out_value, std::uint32_t sending_process = 0);

  /// Scatter primitive in_values[i] to process i
  template <typename T>
  static void scatter(MPI_Comm comm, const std::vector<T>& in_values,
                      T& out_value, std::uint32_t sending_process = 0);

  /// Gather values on one process
  template <typename T>
  static void gather(MPI_Comm comm, const std::vector<T>& in_values,
                     std::vector<T>& out_values,
                     std::uint32_t receiving_process = 0);

  /// Gather strings on one process
  static void gather(MPI_Comm comm, const std::string& in_values,
                     std::vector<std::string>& out_values,
                     std::uint32_t receiving_process = 0);

  /// Gather values from all processes. Same data count from each
  /// process (wrapper for MPI_Allgather)
  template <typename T>
  static void all_gather(MPI_Comm comm, const std::vector<T>& in_values,
                         std::vector<T>& out_values);

  /// Gather values from each process (variable count per process)
  template <typename T>
  static void all_gather(MPI_Comm comm, const std::vector<T>& in_values,
                         std::vector<std::vector<T>>& out_values);

  /// Gather values, one primitive from each process (MPI_Allgather)
  template <typename T>
  static void all_gather(MPI_Comm comm, const T in_value,
                         std::vector<T>& out_values);

  /// Gather values, one primitive from each process (MPI_Allgather).
  /// Specialization for std::string.
  static void all_gather(MPI_Comm comm, const std::string& in_values,
                         std::vector<std::string>& out_values);

  /// Return global max value
  template <typename T>
  static T max(MPI_Comm comm, const T& value);

  /// Return global min value
  template <typename T>
  static T min(MPI_Comm comm, const T& value);

  /// Sum values and return sum
  template <typename T>
  static T sum(MPI_Comm comm, const T& value);

  /// All reduce
  template <typename T, typename X>
  static T all_reduce(MPI_Comm comm, const T& value, X op);

  /// Find global offset (index) (wrapper for MPI_(Ex)Scan with
  /// MPI_SUM as reduction op)
  static std::size_t global_offset(MPI_Comm comm, std::size_t range,
                                   bool exclusive);

  /// Send-receive data between processes (blocking)
  template <typename T>
  static void send_recv(MPI_Comm comm, const std::vector<T>& send_value,
                        std::uint32_t dest, int send_tag,
                        std::vector<T>& recv_value, std::uint32_t source,
                        int recv_tag);

  /// Send-receive data between processes
  template <typename T>
  static void send_recv(MPI_Comm comm, const std::vector<T>& send_value,
                        std::uint32_t dest, std::vector<T>& recv_value,
                        std::uint32_t source);

  /// Return local range for local process, splitting [0, N - 1] into
  /// size() portions of almost equal size
  static std::array<std::int64_t, 2> local_range(MPI_Comm comm, std::int64_t N);

  /// Return local range for given process, splitting [0, N - 1] into
  /// size() portions of almost equal size
  static std::array<std::int64_t, 2> local_range(MPI_Comm comm, int process,
                                                 std::int64_t N);

  /// Return local range for given process, splitting [0, N - 1] into
  /// size() portions of almost equal size
  static std::array<std::int64_t, 2>
  compute_local_range(int process, std::int64_t N, int size);

  /// Return which process owns index (inverse of local_range)
  /// @param[in] size Number of MPI ranks
  /// @param[in] index The index to determine owning rank
  /// @param[in] N Total number of indices
  /// @return The rank of the owning process
  static std::uint32_t index_owner(int size, std::size_t index, std::size_t N);

  template <typename T>
  struct dependent_false : std::false_type
  {
  };

  /// MPI Type
  template <typename T>
  static MPI_Datatype mpi_type()
  {
    static_assert(dependent_false<T>::value, "Unknown MPI type");
    throw std::runtime_error("MPI data type unknown");
    return MPI_CHAR;
  }
};

// Turn off doxygen for these template specialisations
/// @cond
// Specialisations for MPI_Datatypes
template <>
inline MPI_Datatype MPI::mpi_type<float>()
{
  return MPI_FLOAT;
}
template <>
inline MPI_Datatype MPI::mpi_type<double>()
{
  return MPI_DOUBLE;
}
template <>
inline MPI_Datatype MPI::mpi_type<std::complex<double>>()
{
  return MPI_DOUBLE_COMPLEX;
}
template <>
inline MPI_Datatype MPI::mpi_type<short int>()
{
  return MPI_SHORT;
}
template <>
inline MPI_Datatype MPI::mpi_type<int>()
{
  return MPI_INT;
}
template <>
inline MPI_Datatype MPI::mpi_type<unsigned int>()
{
  return MPI_UNSIGNED;
}
template <>
inline MPI_Datatype MPI::mpi_type<long int>()
{
  return MPI_LONG;
}
template <>
inline MPI_Datatype MPI::mpi_type<unsigned long>()
{
  return MPI_UNSIGNED_LONG;
}
template <>
inline MPI_Datatype MPI::mpi_type<long long>()
{
  return MPI_LONG_LONG;
}
template <>
inline MPI_Datatype MPI::mpi_type<unsigned long long>()
{
  return MPI_UNSIGNED_LONG_LONG;
}
/// @endcond
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::broadcast(MPI_Comm comm, std::vector<T>& value,
                             std::uint32_t broadcaster)
{
  // Broadcast cast size
  std::size_t bsize = value.size();
  MPI_Bcast(&bsize, 1, mpi_type<std::size_t>(), broadcaster, comm);

  // Broadcast
  value.resize(bsize);
  MPI_Bcast(const_cast<T*>(value.data()), bsize, mpi_type<T>(), broadcaster,
            comm);
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::broadcast(MPI_Comm comm, T& value, std::uint32_t broadcaster)
{
  MPI_Bcast(&value, 1, mpi_type<T>(), broadcaster, comm);
}
//---------------------------------------------------------------------------
#ifdef DOLFINX_MPI_USE_PUT_GET
template <typename T>
void dolfinx::MPI::all_to_all_common(
    MPI_Comm comm, const std::vector<std::vector<T>>& in_values,
    std::vector<T>& out_values, std::vector<std::int32_t>& local_data_offsets)
{
  const std::size_t comm_size = MPI::size(comm);
  const std::size_t comm_rank = MPI::rank(comm);
  assert(in_values.size() == comm_size);

  // Create a memory area to exchange size information
  // arranged as {offset, size} for each process
  std::vector<int> data_offsets;
  data_offsets.reserve(comm_size * 2);
  int current_offset = 0;
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    data_offsets.push_back(current_offset);
    data_offsets.push_back(in_values[p].size());
    current_offset += data_offsets.back();
  }

  // Flattened data
  std::vector<T> data_send(current_offset);
  // Send offsets to targets that have data to transfer
  std::vector<int> remote_data_offsets(comm_size * 2, 0);
  MPI_Win iwin;
  MPI_Win_create(remote_data_offsets.data(),
                 sizeof(int) * remote_data_offsets.size(), sizeof(int),
                 MPI_INFO_NULL, comm, &iwin);
  MPI_Win_fence(0, iwin);

  for (std::size_t p = 0; p < comm_size; ++p)
  {
    if (in_values[p].size() > 0)
    {
      // Flatten data
      std::copy(in_values[p].begin(), in_values[p].end(),
                data_send.begin() + data_offsets[p * 2]);
      // Meanwhile, send size and offset from remote (if > 0)
      MPI_Put(data_offsets.data() + p * 2, 2, MPI_INT, p, comm_rank * 2, 2,
              MPI_INT, iwin);
    }
  }
  MPI_Win_fence(0, iwin);
  MPI_Win_free(&iwin);

  // Now get the actual data
  // Get local offsets and resize output vector
  local_data_offsets = {0};
  for (std::size_t p = 0; p < comm_size; ++p)
    local_data_offsets.push_back(local_data_offsets.back()
                                 + remote_data_offsets[p * 2 + 1]);
  out_values.resize(local_data_offsets.back());

  MPI_Win Twin;
  MPI_Win_create(data_send.data(), sizeof(T) * data_send.size(), sizeof(T),
                 MPI_INFO_NULL, comm, &Twin);
  MPI_Win_fence(0, Twin);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    const int data_size = remote_data_offsets[p * 2 + 1];
    if (data_size > 0)
      MPI_Get(out_values.data() + local_data_offsets[p], data_size,
              mpi_type<T>(), p, remote_data_offsets[p * 2], data_size,
              mpi_type<T>(), Twin);
  }
  MPI_Win_fence(0, Twin);
  MPI_Win_free(&Twin);
}
#else
// Implementation using MPI_alltoallv
template <typename T>
void dolfinx::MPI::all_to_all_common(
    MPI_Comm comm, const std::vector<std::vector<T>>& in_values,
    std::vector<T>& out_values, std::vector<std::int32_t>& data_offset_recv)
{
  const std::size_t comm_size = MPI::size(comm);

  // Data size per destination
  assert(in_values.size() == comm_size);

  std::vector<int> data_size_send(comm_size);
  std::vector<int> data_offset_send(comm_size + 1, 0);

  for (std::size_t p = 0; p < comm_size; ++p)
  {
    data_size_send[p] = in_values[p].size();
    data_offset_send[p + 1] = data_offset_send[p] + data_size_send[p];
  }

  // Get received data sizes
  std::vector<int> data_size_recv(comm_size);

  MPI_Alltoall(data_size_send.data(), 1, mpi_type<int>(), data_size_recv.data(),
               1, mpi_type<int>(), comm);

  // Pack data and build receive offset
  data_offset_recv.resize(comm_size + 1, 0);
  std::vector<T> data_send(data_offset_send[comm_size]);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    data_offset_recv[p + 1] = data_offset_recv[p] + data_size_recv[p];
    std::copy(in_values[p].begin(), in_values[p].end(),
              data_send.begin() + data_offset_send[p]);
  }

  // Send/receive data
  out_values.resize(data_offset_recv[comm_size]);

  MPI_Alltoallv(data_send.data(), data_size_send.data(),
                data_offset_send.data(), mpi_type<T>(), out_values.data(),
                data_size_recv.data(), data_offset_recv.data(), mpi_type<T>(),
                comm);
}

#endif
//-----------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::all_to_all(MPI_Comm comm,
                              const std::vector<std::vector<T>>& in_values,
                              std::vector<std::vector<T>>& out_values)
{
  std::vector<T> out_vec;
  std::vector<std::int32_t> offsets;
  all_to_all_common(comm, in_values, out_vec, offsets);
  const std::size_t mpi_size = MPI::size(comm);
  out_values.resize(mpi_size);
  for (std::size_t i = 0; i < mpi_size; ++i)
    out_values[i].assign(out_vec.data() + offsets[i],
                         out_vec.data() + offsets[i + 1]);
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::all_to_all(MPI_Comm comm,
                              const std::vector<std::vector<T>>& in_values,
                              std::vector<T>& out_values)
{
  std::vector<std::int32_t> offsets;
  all_to_all_common(comm, in_values, out_values, offsets);
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::scatter(MPI_Comm comm,
                           const std::vector<std::vector<T>>& in_values,
                           std::vector<T>& out_value,
                           std::uint32_t sending_process)
{
  // Scatter number of values to each process
  const std::size_t comm_size = MPI::size(comm);
  std::vector<int> all_num_values;
  if (MPI::rank(comm) == sending_process)
  {
    assert(in_values.size() == comm_size);
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
    const std::size_t n
        = std::accumulate(all_num_values.begin(), all_num_values.end(), 0);
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
               offsets.data(), mpi_type<T>(), out_value.data(), my_num_values,
               mpi_type<T>(), sending_process, comm);
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::scatter(MPI_Comm comm, const std::vector<T>& in_values,
                           T& out_value, std::uint32_t sending_process)
{
  if (MPI::rank(comm) == sending_process)
    assert(in_values.size() == MPI::size(comm));

  MPI_Scatter(const_cast<T*>(in_values.data()), 1, mpi_type<T>(), &out_value, 1,
              mpi_type<T>(), sending_process, comm);
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::gather(MPI_Comm comm, const std::vector<T>& in_values,
                          std::vector<T>& out_values,
                          std::uint32_t receiving_process)
{
  const std::size_t comm_size = MPI::size(comm);

  // Get data size on each process
  std::vector<int> pcounts(comm_size);
  const int local_size = in_values.size();
  MPI_Gather(const_cast<int*>(&local_size), 1, mpi_type<int>(), pcounts.data(),
             1, mpi_type<int>(), receiving_process, comm);

  // Build offsets
  std::vector<int> offsets(comm_size + 1, 0);
  for (std::size_t i = 1; i <= comm_size; ++i)
    offsets[i] = offsets[i - 1] + pcounts[i - 1];

  const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
  out_values.resize(n);
  MPI_Gatherv(const_cast<T*>(in_values.data()), in_values.size(), mpi_type<T>(),
              out_values.data(), pcounts.data(), offsets.data(), mpi_type<T>(),
              receiving_process, comm);
}
//---------------------------------------------------------------------------
inline void dolfinx::MPI::gather(MPI_Comm comm, const std::string& in_values,
                                 std::vector<std::string>& out_values,
                                 std::uint32_t receiving_process)
{
  const std::size_t comm_size = MPI::size(comm);

  // Get data size on each process
  std::vector<int> pcounts(comm_size);
  int local_size = in_values.size();
  MPI_Gather(&local_size, 1, MPI_INT, pcounts.data(), 1, MPI_INT,
             receiving_process, comm);

  // Build offsets
  std::vector<int> offsets(comm_size + 1, 0);
  for (std::size_t i = 1; i <= comm_size; ++i)
    offsets[i] = offsets[i - 1] + pcounts[i - 1];

  // Gather
  const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
  std::vector<char> _out(n);
  MPI_Gatherv(const_cast<char*>(in_values.data()), in_values.size(), MPI_CHAR,
              _out.data(), pcounts.data(), offsets.data(), MPI_CHAR,
              receiving_process, comm);

  // Rebuild
  out_values.resize(comm_size);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    out_values[p]
        = std::string(_out.begin() + offsets[p], _out.begin() + offsets[p + 1]);
  }
}
//---------------------------------------------------------------------------
inline void dolfinx::MPI::all_gather(MPI_Comm comm,
                                     const std::string& in_values,
                                     std::vector<std::string>& out_values)
{
  const std::size_t comm_size = MPI::size(comm);

  // Get data size on each process
  std::vector<int> pcounts(comm_size);
  int local_size = in_values.size();
  MPI_Allgather(&local_size, 1, MPI_INT, pcounts.data(), 1, MPI_INT, comm);

  // Build offsets
  std::vector<int> offsets(comm_size + 1, 0);
  for (std::size_t i = 1; i <= comm_size; ++i)
    offsets[i] = offsets[i - 1] + pcounts[i - 1];

  // Gather
  const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
  std::vector<char> _out(n);
  MPI_Allgatherv(const_cast<char*>(in_values.data()), in_values.size(),
                 MPI_CHAR, _out.data(), pcounts.data(), offsets.data(),
                 MPI_CHAR, comm);

  // Rebuild
  out_values.resize(comm_size);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    out_values[p]
        = std::string(_out.begin() + offsets[p], _out.begin() + offsets[p + 1]);
  }
}
//-------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::all_gather(MPI_Comm comm, const std::vector<T>& in_values,
                              std::vector<T>& out_values)
{
  out_values.resize(in_values.size() * MPI::size(comm));
  MPI_Allgather(const_cast<T*>(in_values.data()), in_values.size(),
                mpi_type<T>(), out_values.data(), in_values.size(),
                mpi_type<T>(), comm);
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::all_gather(MPI_Comm comm, const std::vector<T>& in_values,
                              std::vector<std::vector<T>>& out_values)
{
  const std::size_t comm_size = MPI::size(comm);

  // Get data size on each process
  std::vector<int> pcounts;
  const int local_size = in_values.size();
  MPI::all_gather(comm, local_size, pcounts);
  assert(pcounts.size() == comm_size);

  // Build offsets
  std::vector<int> offsets(comm_size + 1, 0);
  for (std::size_t i = 1; i <= comm_size; ++i)
    offsets[i] = offsets[i - 1] + pcounts[i - 1];

  // Gather data
  const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
  std::vector<T> recvbuf(n);
  MPI_Allgatherv(const_cast<T*>(in_values.data()), in_values.size(),
                 mpi_type<T>(), recvbuf.data(), pcounts.data(), offsets.data(),
                 mpi_type<T>(), comm);

  // Repack data
  out_values.resize(comm_size);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    out_values[p].resize(pcounts[p]);
    for (int i = 0; i < pcounts[p]; ++i)
      out_values[p][i] = recvbuf[offsets[p] + i];
  }
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::all_gather(MPI_Comm comm, const T in_value,
                              std::vector<T>& out_values)
{
  out_values.resize(MPI::size(comm));
  MPI_Allgather(const_cast<T*>(&in_value), 1, mpi_type<T>(), out_values.data(),
                1, mpi_type<T>(), comm);
}
//---------------------------------------------------------------------------
template <typename T, typename X>
T dolfinx::MPI::all_reduce(MPI_Comm comm, const T& value, X op)
{
  T out;
  MPI_Allreduce(const_cast<T*>(&value), &out, 1, mpi_type<T>(), op, comm);
  return out;
}
//-----------------------------------------------------------------------------
template <typename T>
T dolfinx::MPI::max(MPI_Comm comm, const T& value)
{
  // Enforce cast to MPI_Op; this is needed because template dispatch may
  // not recognize this is possible, e.g. C-enum to std::uint32_t in SGI MPT
  MPI_Op op = static_cast<MPI_Op>(MPI_MAX);
  return all_reduce(comm, value, op);
}
//---------------------------------------------------------------------------
template <typename T>
T dolfinx::MPI::min(MPI_Comm comm, const T& value)
{
  // Enforce cast to MPI_Op; this is needed because template dispatch may
  // not recognize this is possible, e.g. C-enum to std::uint32_t in SGI MPT
  MPI_Op op = static_cast<MPI_Op>(MPI_MIN);
  return all_reduce(comm, value, op);
}
//---------------------------------------------------------------------------
template <typename T>
T dolfinx::MPI::sum(MPI_Comm comm, const T& value)
{
  // Enforce cast to MPI_Op; this is needed because template dispatch may
  // not recognize this is possible, e.g. C-enum to std::uint32_t in SGI MPT
  MPI_Op op = static_cast<MPI_Op>(MPI_SUM);
  return all_reduce(comm, value, op);
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::send_recv(MPI_Comm comm, const std::vector<T>& send_value,
                             std::uint32_t dest, int send_tag,
                             std::vector<T>& recv_value, std::uint32_t source,
                             int recv_tag)
{
  std::size_t send_size = send_value.size();
  std::size_t recv_size = 0;
  MPI_Status mpi_status;
  MPI_Sendrecv(&send_size, 1, mpi_type<std::size_t>(), dest, send_tag,
               &recv_size, 1, mpi_type<std::size_t>(), source, recv_tag, comm,
               &mpi_status);

  recv_value.resize(recv_size);
  MPI_Sendrecv(const_cast<T*>(send_value.data()), send_value.size(),
               mpi_type<T>(), dest, send_tag, recv_value.data(), recv_size,
               mpi_type<T>(), source, recv_tag, comm, &mpi_status);
}
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::send_recv(MPI_Comm comm, const std::vector<T>& send_value,
                             std::uint32_t dest, std::vector<T>& recv_value,
                             std::uint32_t source)
{
  MPI::send_recv(comm, send_value, dest, 0, recv_value, source, 0);
}
//-----------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::neighbor_all_to_all(MPI_Comm neighbor_comm,
                                       const std::vector<int>& send_offsets,
                                       const std::vector<T>& send_data,
                                       std::vector<int>& recv_offsets,
                                       std::vector<T>& recv_data)
{
  assert((int)send_data.size() == send_offsets.back());
  assert(send_offsets[0] == 0);

  // Get receive sizes
  std::vector<int> send_sizes(send_offsets.size() - 1, 0);
  std::vector<int> recv_sizes(send_sizes.size());
  for (std::size_t i = 0; i < send_sizes.size(); ++i)
    send_sizes[i] = send_offsets[i + 1] - send_offsets[i];
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI::mpi_type<int>(),
                        recv_sizes.data(), 1, MPI::mpi_type<int>(),
                        neighbor_comm);

  // Work out recv offsets
  recv_offsets.resize(recv_sizes.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   recv_offsets.begin() + 1);
  recv_data.resize(recv_offsets.back());

  MPI_Neighbor_alltoallv(
      send_data.data(), send_sizes.data(), send_offsets.data(),
      MPI::mpi_type<T>(), recv_data.data(), recv_sizes.data(),
      recv_offsets.data(), MPI::mpi_type<T>(), neighbor_comm);
}

} // namespace dolfinx
