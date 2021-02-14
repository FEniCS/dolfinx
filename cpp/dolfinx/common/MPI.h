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
#include <dolfinx/graph/AdjacencyList.h>
#include <iostream>
#include <numeric>
#include <set>
#include <tuple>
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
    explicit Comm(MPI_Comm comm, bool duplicate = true);

    /// Copy constructor
    Comm(const Comm& comm);

    /// Move constructor
    Comm(Comm&& comm) noexcept;

    // Disable copy assignment operator
    Comm& operator=(const Comm& comm) = delete;

    /// Move assignment operator
    Comm& operator=(Comm&& comm) noexcept;

    /// Destructor (frees wrapped communicator)
    ~Comm();

    /// Return the underlying MPI_Comm object
    MPI_Comm comm() const;

  private:
    // MPI communicator
    MPI_Comm _comm;
  };

  /// Return process rank for the communicator
  static int rank(MPI_Comm comm);

  /// Return size of the group (number of processes) associated with the
  /// communicator
  static int size(MPI_Comm comm);

  /// Send in_values[p0] to process p0 and receive values from process
  /// p1 in out_values[p1]
  template <typename T>
  static graph::AdjacencyList<T>
  all_to_all(MPI_Comm comm, const graph::AdjacencyList<T>& send_data);

  /// @todo Experimental. Maybe be moved or removed.
  ///
  /// Compute communication graph edges. The caller provides edges that
  /// it can define, and will receive edges to it that are defined by
  /// other ranks.
  ///
  /// @note This function involves global communication
  ///
  /// @param[in] comm The MPI communicator
  /// @param[in] edges Communication edges between the caller and the
  ///   ranks in @p edges.
  /// @return Ranks that have defined edges from them to this rank
  static std::vector<int> compute_graph_edges(MPI_Comm comm,
                                              const std::set<int>& edges);

  /// Neighborhood all-to-all. Send data to neighbors.
  /// Send in_values[n0] to neighbor process n0 and receive values from neighbor
  /// process n1 in out_values[n1]
  template <typename T>
  static graph::AdjacencyList<T>
  neighbor_all_to_all(MPI_Comm neighbor_comm,
                      const graph::AdjacencyList<T>& send_data);

  /// @todo Clarify directions
  ///
  /// Return list of neighbors for a neighborhood communicator
  /// @param[in] neighbor_comm Neighborhood communicator
  /// @return source ranks, destination ranks
  static std::tuple<std::vector<int>, std::vector<int>>
  neighbors(MPI_Comm neighbor_comm);

  /// Find global offset (index) (wrapper for MPI_(Ex)Scan with MPI_SUM
  /// as reduction op)
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
graph::AdjacencyList<T>
dolfinx::MPI::all_to_all(MPI_Comm comm,
                         const graph::AdjacencyList<T>& send_data)
{
  const std::vector<std::int32_t>& send_offsets = send_data.offsets();
  const std::vector<T>& values_in = send_data.array();

  const int comm_size = MPI::size(comm);
  assert(send_data.num_nodes() == comm_size);

  // Data size per destination rank
  std::vector<int> send_size(comm_size);
  std::adjacent_difference(std::next(send_offsets.begin(), +1),
                           send_offsets.end(), send_size.begin());

  // Get received data sizes from each rank
  std::vector<int> recv_size(comm_size);
  MPI_Alltoall(send_size.data(), 1, mpi_type<int>(), recv_size.data(), 1,
               mpi_type<int>(), comm);

  // Compute receive offset
  std::vector<std::int32_t> recv_offset(comm_size + 1);
  recv_offset[0] = 0;
  std::partial_sum(recv_size.begin(), recv_size.end(),
                   std::next(recv_offset.begin()));

  // Send/receive data
  std::vector<T> recv_values(recv_offset[comm_size]);
  MPI_Alltoallv(values_in.data(), send_size.data(), send_offsets.data(),
                mpi_type<T>(), recv_values.data(), recv_size.data(),
                recv_offset.data(), mpi_type<T>(), comm);

  return graph::AdjacencyList<T>(std::move(recv_values),
                                 std::move(recv_offset));
}
//-----------------------------------------------------------------------------
template <typename T>
graph::AdjacencyList<T>
dolfinx::MPI::neighbor_all_to_all(MPI_Comm neighbor_comm,
                                  const graph::AdjacencyList<T>& send_data)
{
  // Get neighbor processes
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbor_comm, &indegree, &outdegree,
                                 &weighted);

  // Get receive sizes
  std::vector<int> send_sizes(outdegree, 0);
  std::vector<int> recv_sizes(indegree);
  std::adjacent_difference(std::next(send_data.offsets().begin()),
                           send_data.offsets().end(), send_sizes.begin());
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI::mpi_type<int>(),
                        recv_sizes.data(), 1, MPI::mpi_type<int>(),
                        neighbor_comm);

  // Work out recv offsets
  std::vector<int> recv_offsets(recv_sizes.size() + 1);
  recv_offsets[0] = 0;
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_offsets.begin(), 1));

  std::vector<T> recv_data(recv_offsets[recv_offsets.size() - 1]);
  MPI_Neighbor_alltoallv(
      send_data.array().data(), send_sizes.data(), send_data.offsets().data(),
      MPI::mpi_type<T>(), recv_data.data(), recv_sizes.data(),
      recv_offsets.data(), MPI::mpi_type<T>(), neighbor_comm);

  return graph::AdjacencyList<T>(std::move(recv_data), std::move(recv_offsets));
}
//---------------------------------------------------------------------------

} // namespace dolfinx
