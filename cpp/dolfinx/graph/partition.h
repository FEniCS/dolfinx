// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <mpi.h>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::graph
{

/// Signature of functions for computing the parallel partitioning of a
/// distributed graph
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across
/// @param[in] nparts Number of partitions to divide graph nodes into
/// @param[in] local_graph Node connectivity graph
/// @param[in] num_ghost_nodes Number of graph nodes appearing in @p
/// local_graph that are owned on other processes
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution
/// @return Destination rank for each input node
using partition_fn = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const AdjacencyList<std::int64_t>&, std::int32_t, bool)>;

/// Partition graph across processes using the default graph partitioner
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across
/// @param[in] nparts Number of partitions to divide graph nodes into
/// @param[in] local_graph Node connectivity graph
/// @param[in] num_ghost_nodes Number of graph nodes appearing in @p
/// local_graph that are owned on other processes
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution
/// @return Destination rank for each input node
AdjacencyList<std::int32_t>
partition_graph(MPI_Comm comm, int nparts,
                const AdjacencyList<std::int64_t>& local_graph,
                std::int32_t num_ghost_nodes, bool ghosting);

/// Tools for distributed graphs
///
/// @todo Add a function that sends data to the 'owner'
namespace build
{
/// Distribute adjacency list nodes to destination ranks. The global
/// index of each node is assumed to be the local index plus the
/// offset for this rank.
///
/// @param[in] comm MPI Communicator
/// @param[in] list The adjacency list to distribute
/// @param[in] destinations Destination ranks for the ith node in the
/// adjacency list
/// @return Adjacency list for this process, array of source ranks for
/// each node in the adjacency list, and the original global index for
/// each node.
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
distribute(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& list,
           const graph::AdjacencyList<std::int32_t>& destinations);

/// Compute ghost indices in a global IndexMap space, from a list of arbitrary
/// global indices, where the ghosts are at the end of the list, and their
/// owning processes are known.
/// @param[in] comm MPI communicator
/// @param[in] global_indices List of arbitrary global indices, ghosts at end
/// @param[in] ghost_owners List of owning processes of the ghost indices
/// @return Indexing of ghosts in a global space starting from 0 on process 0
std::vector<std::int64_t>
compute_ghost_indices(MPI_Comm comm,
                      const xtl::span<const std::int64_t>& global_indices,
                      const xtl::span<const int>& ghost_owners);

/// Distribute rows of a rectangular data array to process ranks where
/// it it required
///
/// @param[in] comm The MPI communicator
/// @param[in] indices Global indices of the data (rows) required by
/// this process
/// @param[in] x Data on this process which may be distributed (by row).
/// The global index for the `[0, ..., n)` local rows is assumed to be
/// the local index plus the offset for this rank. Layout is row-major.
/// @param[in] shape1 The number of columns of the data array
/// @return The data for each index in @p indices (row-major storage)
/// @pre `shape1 > 0`
template <typename T>
std::vector<T> distribute_data(MPI_Comm comm,
                               const xtl::span<const std::int64_t>& indices,
                               const xtl::span<const T>& x, int shape1);

/// Given an adjacency list with global, possibly non-contiguous, link
/// indices and a local adjacency list with contiguous link indices
/// starting from zero, compute a local-to-global map for the links.
/// Both adjacency lists must have the same shape.
///
/// @param[in] global Adjacency list with global link indices
/// @param[in] local Adjacency list with local, contiguous link indices
/// @return Map from local index to global index, which if applied to
/// the local adjacency list indices would yield the global adjacency
/// list
std::vector<std::int64_t>
compute_local_to_global_links(const graph::AdjacencyList<std::int64_t>& global,
                              const graph::AdjacencyList<std::int32_t>& local);

/// Compute a local0-to-local1 map from two local-to-global maps with
/// common global indices
///
/// @param[in] local0_to_global Map from local0 indices to global
/// indices
/// @param[in] local1_to_global Map from local1 indices to global
/// indices
/// @return Map from local0 indices to local1 indices
std::vector<std::int32_t>
compute_local_to_local(const xtl::span<const std::int64_t>& local0_to_global,
                       const xtl::span<const std::int64_t>& local1_to_global);
} // namespace build

//---------------------------------------------------------------------------
// Implementation
//---------------------------------------------------------------------------
template <typename T>
std::vector<T>
build::distribute_data(MPI_Comm comm,
                       const xtl::span<const std::int64_t>& indices,
                       const xtl::span<const T>& x, int shape1)
{
  common::Timer timer("Fetch float data from remote processes");
  assert(shape1 > 0);

  assert(x.size() % shape1 == 0);
  const std::int64_t shape0 = x.size() / shape1;
  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  std::vector<std::int64_t> global_sizes(size);
  MPI_Allgather(&shape0, 1, MPI_INT64_T, global_sizes.data(), 1, MPI_INT64_T,
                comm);
  std::vector<std::int64_t> global_offsets(size + 1, 0);
  std::partial_sum(global_sizes.begin(), global_sizes.end(),
                   global_offsets.begin() + 1);

  // Build index data requests
  std::vector<int> number_index_send(size, 0);
  std::vector<int> index_owner(indices.size());
  std::vector<int> index_order(indices.size());
  std::iota(index_order.begin(), index_order.end(), 0);
  std::sort(index_order.begin(), index_order.end(),
            [&indices](int a, int b) { return (indices[a] < indices[b]); });

  int p = 0;
  for (std::size_t i = 0; i < index_order.size(); ++i)
  {
    int j = index_order[i];
    while (indices[j] >= global_offsets[p + 1])
      ++p;
    index_owner[j] = p;
    number_index_send[p]++;
  }

  // Compute send displacements
  std::vector<int> disp_index_send(size + 1, 0);
  std::partial_sum(number_index_send.begin(), number_index_send.end(),
                   disp_index_send.begin() + 1);

  // Pack global index send data
  std::vector<std::int64_t> indices_send(disp_index_send.back());
  std::vector<int> disp_tmp = disp_index_send;
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    const int owner = index_owner[i];
    indices_send[disp_tmp[owner]++] = indices[i];
  }

  // Send/receive number of indices to communicate to each process
  std::vector<int> number_index_recv(size);
  MPI_Alltoall(number_index_send.data(), 1, MPI_INT, number_index_recv.data(),
               1, MPI_INT, comm);

  // Compute receive displacements
  std::vector<int> disp_index_recv(size + 1, 0);
  std::partial_sum(number_index_recv.begin(), number_index_recv.end(),
                   disp_index_recv.begin() + 1);

  // Send/receive global indices
  std::vector<std::int64_t> indices_recv(disp_index_recv.back());
  MPI_Alltoallv(indices_send.data(), number_index_send.data(),
                disp_index_send.data(), MPI_INT64_T, indices_recv.data(),
                number_index_recv.data(), disp_index_recv.data(), MPI_INT64_T,
                comm);

  // Pack point data to send back (transpose)
  std::vector<T> x_return(indices_recv.size() * shape1);
  for (int p = 0; p < size; ++p)
  {
    for (int i = disp_index_recv[p]; i < disp_index_recv[p + 1]; ++i)
    {
      const std::int32_t index_local = indices_recv[i] - global_offsets[rank];
      assert(index_local >= 0);
      std::copy_n(std::next(x.cbegin(), shape1 * index_local), shape1,
                  std::next(x_return.begin(), shape1 * i));
    }
  }

  MPI_Datatype compound_type;
  MPI_Type_contiguous(shape1, dolfinx::MPI::mpi_type<T>(), &compound_type);
  MPI_Type_commit(&compound_type);

  // Send back point data
  std::vector<T> my_x(disp_index_send.back() * shape1);
  MPI_Alltoallv(x_return.data(), number_index_recv.data(),
                disp_index_recv.data(), compound_type, my_x.data(),
                number_index_send.data(), disp_index_send.data(), compound_type,
                comm);
  MPI_Type_free(&compound_type);

  return my_x;
}

} // namespace dolfinx::graph
