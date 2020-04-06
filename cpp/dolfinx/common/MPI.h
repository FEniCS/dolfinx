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
  static int rank(MPI_Comm comm);

  /// Return size of the group (number of processes) associated with the
  /// communicator
  static int size(MPI_Comm comm);

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

  /// Find global offset (index) (wrapper for MPI_(Ex)Scan with
  /// MPI_SUM as reduction op)
  static std::size_t global_offset(MPI_Comm comm, std::size_t range,
                                   bool exclusive);

  /// Return local range for given process, splitting [0, N - 1] into
  /// size() portions of almost equal size
  static std::array<std::int64_t, 2> local_range(int process, std::int64_t N,
                                                 int size);

  /// Return which process owns index (inverse of local_range)
  /// @param[in] size Number of MPI ranks
  /// @param[in] index The index to determine owning rank
  /// @param[in] N Total number of indices
  /// @return The rank of the owning process
  static int index_owner(int size, std::size_t index, std::size_t N);

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
template <>
inline MPI_Datatype MPI::mpi_type<bool>()
{
  return MPI_C_BOOL;
}
/// @endcond
//---------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::all_to_all_common(
    MPI_Comm comm, const std::vector<std::vector<T>>& in_values,
    std::vector<T>& out_values, std::vector<std::int32_t>& data_offset_recv)
{
  const int comm_size = MPI::size(comm);

  // Data size per destination
  assert((int)in_values.size() == comm_size);
  std::vector<int> data_size_send(comm_size);
  std::vector<int> data_offset_send(comm_size + 1, 0);
  for (int p = 0; p < comm_size; ++p)
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
  for (int p = 0; p < comm_size; ++p)
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
//-----------------------------------------------------------------------------
template <typename T>
void dolfinx::MPI::all_to_all(MPI_Comm comm,
                              const std::vector<std::vector<T>>& in_values,
                              std::vector<std::vector<T>>& out_values)
{
  std::vector<T> out_vec;
  std::vector<std::int32_t> offsets;
  all_to_all_common(comm, in_values, out_vec, offsets);
  const int mpi_size = MPI::size(comm);
  out_values.resize(mpi_size);
  for (int i = 0; i < mpi_size; ++i)
  {
    out_values[i].assign(out_vec.data() + offsets[i],
                         out_vec.data() + offsets[i + 1]);
  }
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
//---------------------------------------------------------------------------

} // namespace dolfinx
