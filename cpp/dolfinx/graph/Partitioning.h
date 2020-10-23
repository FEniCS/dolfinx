// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <mpi.h>
#include <set>
#include <utility>
#include <vector>

namespace dolfinx::graph
{

/// Tools for distributed graphs
///
/// TODO: Add a function that sends data (Eigen arrays) to the 'owner'

namespace Partitioning
{
/// @todo Return the list of neighbor processes which is computed
/// internally
///
/// Compute new, contiguous global indices from a collection of
/// global, possibly globally non-contiguous, indices and assign
/// process ownership to the new global indices such that the global
/// index of owned indices increases with increasing MPI rank.
///
/// @param[in] comm The communicator across which the indices are
///   distributed
/// @param[in] global_indices Global indices on this process. Some
///   global indices may also be on other processes
/// @param[in] shared_indices Vector that is true for indices that may
///   also be in other process. Size is the same as @p global_indices.
/// @return {Local (old, from local_to_global) -> local (new) indices,
///   global indices for ghosts of this process}. The new indices are
///   [0, ..., N), with [0, ..., n0) being owned. The new global index
///   for an owned index is n_global = n + offset, where offset is
///   computed from a process scan. Indices [n0, ..., N) are owned by
///   a remote process and the ghosts return vector maps [n0, ..., N)
///   to global indices.
std::tuple<std::vector<std::int32_t>, std::vector<std::int64_t>,
           std::vector<int>>
reorder_global_indices(MPI_Comm comm,
                       const std::vector<std::int64_t>& global_indices,
                       const std::vector<bool>& shared_indices);

/// Compute a local AdjacencyList list with contiguous indices from an
/// AdjacencyList that may have non-contiguous data
///
/// @param[in] list Adjacency list with links that might not have
///   contiguous numdering
/// @return Adjacency list with contiguous ordering [0, 1, ..., n),
///   and a map from local indices in the returned Adjacency list to
///   the global indices in @p list
std::pair<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>>
create_local_adjacency_list(const graph::AdjacencyList<std::int64_t>& list);

/// Build a distributed AdjacencyList list with re-numbered links from
/// an AdjacencyList that may have non-contiguous data. The
/// distribution of the AdjacencyList nodes is unchanged.
///
/// @param[in] comm MPI communicator
/// @param[in] list_local Local adjacency list, with contiguous link
///   indices
/// @param[in] local_to_global_links Local-to-global map for links in
///   the local adjacency list
/// @param[in] shared_links Try for possible shared links
std::tuple<graph::AdjacencyList<std::int32_t>, common::IndexMap>
create_distributed_adjacency_list(
    MPI_Comm comm, const graph::AdjacencyList<std::int32_t>& list_local,
    const std::vector<std::int64_t>& local_to_global_links,
    const std::vector<bool>& shared_links);

/// Distribute adjacency list nodes to destination ranks. The global
/// index of each node is assumed to be the local index plus the
/// offset for this rank.
///
/// @param[in] comm MPI Communicator
/// @param[in] list The adjacency list to distribute
/// @param[in] destinations Destination ranks for the ith node in the
///   adjacency list
/// @return Adjacency list for this process, array of source ranks for
///   each node in the adjacency list, and the original global index
///   for each node.
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
                      const std::vector<std::int64_t>& global_indices,
                      const std::vector<int>& ghost_owners);

/// Distribute data to process ranks where it it required
///
/// @param[in] comm The MPI communicator
/// @param[in] indices Global indices of the data required by this
///   process
/// @param[in] x Data on this process which may be distributed (by
///   row). The global index for the [0, ..., n) local rows is assumed
///   to be the local index plus the offset for this rank
/// @return The data for each index in @p indices
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
distribute_data(MPI_Comm comm, const std::vector<std::int64_t>& indices,
                const Eigen::Ref<const Eigen::Array<
                    T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& x);

/// Given an adjacency list with global, possibly non-contiguous, link
/// indices and a local adjacency list with contiguous link indices
/// starting from zero, compute a local-to-global map for the links.
/// Both adjacency lists must have the same shape.
///
/// @param[in] global Adjacency list with global link indices
/// @param[in] local Adjacency list with local, contiguous link
///   indices
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
///   indices
/// @param[in] local1_to_global Map from local1 indices to global
///   indices
/// @return Map from local0 indices to local1 indices
std::vector<std::int32_t>
compute_local_to_local(const std::vector<std::int64_t>& local0_to_global,
                       const std::vector<std::int64_t>& local1_to_global);
} // namespace Partitioning

//---------------------------------------------------------------------------
// Implementation
//---------------------------------------------------------------------------
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Partitioning::distribute_data(
    MPI_Comm comm, const std::vector<std::int64_t>& indices,
    const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x)
{
  common::Timer timer("Fetch float data from remote processes");

  const std::int64_t num_points_local = x.rows();
  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  std::vector<std::int64_t> global_sizes(size);
  MPI_Allgather(&num_points_local, 1, MPI_INT64_T, global_sizes.data(), 1,
                MPI_INT64_T, comm);
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

  const int item_size = x.cols();
  assert(item_size != 0);
  // Pack point data to send back (transpose)
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_return(
      indices_recv.size(), item_size);
  for (int p = 0; p < size; ++p)
  {
    for (int i = disp_index_recv[p]; i < disp_index_recv[p + 1]; ++i)
    {
      const std::int32_t index_local = indices_recv[i] - global_offsets[rank];
      assert(index_local >= 0);
      x_return.row(i) = x.row(index_local);
    }
  }

  MPI_Datatype compound_type;
  MPI_Type_contiguous(item_size, dolfinx::MPI::mpi_type<T>(), &compound_type);
  MPI_Type_commit(&compound_type);

  // Send back point data
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> my_x(
      disp_index_send.back(), item_size);
  MPI_Alltoallv(x_return.data(), number_index_recv.data(),
                disp_index_recv.data(), compound_type, my_x.data(),
                number_index_send.data(), disp_index_send.data(), compound_type,
                comm);

  return my_x;
}

} // namespace dolfinx::graph
