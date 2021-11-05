// Copyright (C) 2007-2014 Magnus Vikstrøm and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
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

/// MPI support functionality
namespace dolfinx::MPI
{

/// A duplicate MPI communicator and manage lifetime of the
/// communicator
class Comm
{
public:
  /// Duplicate communicator and wrap duplicate
  explicit Comm(MPI_Comm comm, bool duplicate = true);

  /// Copy constructor
  Comm(const Comm& comm) noexcept;

  /// Move constructor
  Comm(Comm&& comm) noexcept;

  // Disable copy assignment operator
  Comm& operator=(const Comm& comm) = delete;

  /// Move assignment operator
  Comm& operator=(Comm&& comm) noexcept;

  /// Destructor (frees wrapped communicator)
  ~Comm();

  /// Return the underlying MPI_Comm object
  MPI_Comm comm() const noexcept;

private:
  // MPI communicator
  MPI_Comm _comm;
};

/// Return process rank for the communicator
int rank(MPI_Comm comm);

/// Return size of the group (number of processes) associated with the
/// communicator
int size(MPI_Comm comm);

/// Send in_values[p0] to process p0 and receive values from process
/// p1 in out_values[p1]
template <typename T>
graph::AdjacencyList<T> all_to_all(MPI_Comm comm,
                                   const graph::AdjacencyList<T>& send_data);

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
std::vector<int> compute_graph_edges(MPI_Comm comm, const std::set<int>& edges);

/// Neighborhood all-to-all. Send data to neighbors.
/// Send in_values[n0] to neighbor process n0 and receive values from neighbor
/// process n1 in out_values[n1]
template <typename T>
graph::AdjacencyList<T>
neighbor_all_to_all(MPI_Comm neighbor_comm,
                    const graph::AdjacencyList<T>& send_data);

/// Return list of neighbors (sources and and destination) for a
/// neighborhood communicator
/// @param[in] comm Neighborhood communicator
/// @return source ranks, destination ranks
std::array<std::vector<int>, 2> neighbors(MPI_Comm comm);

/// Return local range for given process, splitting [0, N - 1] into
/// size() portions of almost equal size
/// @param[in] rank MPI rank of the caller
/// @param[in] N The value to partition
/// @param[in] size The number of MPI ranks across which to partition
/// `N`
constexpr std::array<std::int64_t, 2> local_range(int rank, std::int64_t N,
                                                  int size)
{
  assert(rank >= 0);
  assert(N >= 0);
  assert(size > 0);

  // Compute number of items per rank and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  if (rank < r)
    return {{rank * (n + 1), rank * (n + 1) + n + 1}};
  else
    return {{rank * n + r, rank * n + r + n}};
}

/// Return which process owns index (inverse of local_range)
/// @param[in] size Number of MPI ranks
/// @param[in] index The index to determine owning rank
/// @param[in] N Total number of indices
/// @return The rank of the owning process
constexpr int index_owner(int size, std::size_t index, std::size_t N)
{
  assert(index < N);

  // Compute number of items per rank and remainder
  const std::size_t n = N / size;
  const std::size_t r = N % size;

  // First r ranks own n + 1 indices
  if (index < r * (n + 1))
    return index / (n + 1);

  // Remaining ranks own n indices
  return r + (index - r * (n + 1)) / n;
}

template <typename T>
struct dependent_false : std::false_type
{
};

/// MPI Type
template <typename T>
constexpr MPI_Datatype mpi_type()
{
  if constexpr (std::is_same<T, float>::value)
    return MPI_FLOAT;
  else if constexpr (std::is_same<T, double>::value)
    return MPI_DOUBLE;
  else if constexpr (std::is_same<T, std::complex<double>>::value)
    return MPI_DOUBLE_COMPLEX;
  else if constexpr (std::is_same<T, short int>::value)
    return MPI_SHORT;
  else if constexpr (std::is_same<T, int>::value)
    return MPI_INT;
  else if constexpr (std::is_same<T, unsigned int>::value)
    return MPI_UNSIGNED;
  else if constexpr (std::is_same<T, long int>::value)
    return MPI_LONG;
  else if constexpr (std::is_same<T, unsigned long>::value)
    return MPI_UNSIGNED_LONG;
  else if constexpr (std::is_same<T, long long>::value)
    return MPI_LONG_LONG;
  else if constexpr (std::is_same<T, unsigned long long>::value)
    return MPI_UNSIGNED_LONG_LONG;
  else if constexpr (std::is_same<T, bool>::value)
    return MPI_C_BOOL;
}

//---------------------------------------------------------------------------
template <typename T>
graph::AdjacencyList<T> all_to_all(MPI_Comm comm,
                                   const graph::AdjacencyList<T>& send_data)
{
  const std::vector<std::int32_t>& send_offsets = send_data.offsets();
  const std::vector<T>& values_in = send_data.array();

  const int comm_size = MPI::size(comm);
  assert(send_data.num_nodes() == comm_size);

  // Data size per destination rank
  std::vector<int> send_size(comm_size);
  std::adjacent_difference(std::next(send_offsets.begin()), send_offsets.end(),
                           send_size.begin());

  // Get received data sizes from each rank
  std::vector<int> recv_size(comm_size);
  MPI_Alltoall(send_size.data(), 1, mpi_type<int>(), recv_size.data(), 1,
               mpi_type<int>(), comm);

  // Compute receive offset
  std::vector<std::int32_t> recv_offset(comm_size + 1, 0);
  std::partial_sum(recv_size.begin(), recv_size.end(),
                   std::next(recv_offset.begin()));

  // Send/receive data
  std::vector<T> recv_values(recv_offset.back());
  MPI_Alltoallv(values_in.data(), send_size.data(), send_offsets.data(),
                mpi_type<T>(), recv_values.data(), recv_size.data(),
                recv_offset.data(), mpi_type<T>(), comm);

  return graph::AdjacencyList<T>(std::move(recv_values),
                                 std::move(recv_offset));
}
//-----------------------------------------------------------------------------
template <typename T>
graph::AdjacencyList<T>
neighbor_all_to_all(MPI_Comm neighbor_comm,
                    const graph::AdjacencyList<T>& send_data)
{
  // Get neighbor processes
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbor_comm, &indegree, &outdegree,
                                 &weighted);

  // Allocate memory (add '1' to handle empty case as OpenMPI fails for
  // null pointers
  std::vector<int> send_sizes(outdegree + 1, 0);
  std::vector<int> recv_sizes(indegree + 1);
  std::adjacent_difference(std::next(send_data.offsets().begin()),
                           send_data.offsets().end(), send_sizes.begin());
  // Get receive sizes
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI::mpi_type<int>(),
                        recv_sizes.data(), 1, MPI::mpi_type<int>(),
                        neighbor_comm);

  // Work out recv offsets. Note use of std::prev to handle OpenMPI
  // issue mentioned above
  std::vector<int> recv_offsets(indegree + 1);
  recv_offsets[0] = 0;
  std::partial_sum(recv_sizes.begin(), std::prev(recv_sizes.end()),
                   std::next(recv_offsets.begin(), 1));

  std::vector<T> recv_data(recv_offsets[recv_offsets.size() - 1]);
  MPI_Neighbor_alltoallv(
      send_data.array().data(), send_sizes.data(), send_data.offsets().data(),
      MPI::mpi_type<T>(), recv_data.data(), recv_sizes.data(),
      recv_offsets.data(), MPI::mpi_type<T>(), neighbor_comm);

  return graph::AdjacencyList<T>(std::move(recv_data), std::move(recv_offsets));
}
//---------------------------------------------------------------------------

} // namespace dolfinx::MPI
