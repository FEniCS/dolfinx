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
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <mpi.h>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

#include <iostream>

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
/// it ia required
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
namespace impl
{

/// @brief Send data to 'post office' rank (by first index in each row)
/// @param[in] comm MPI communicator
/// @param[in] x Data to distribute (2D, row-major layout)
/// @param[in] shape1 Number of columns for `x`
/// @param[in] shape0_global Number of global rows for  `x`
/// @param[in] rank_offset The rank offset such that global index of
/// local row `i` in `x` is `rank_offset + i`.
/// @returns (0) global indices of my post office data and (1) the data
/// (row-major). It **does not** include rows that are in `x`.
template <typename T>
std::pair<std::vector<std::int64_t>, std::vector<T>>
send_to_postoffice(MPI_Comm comm, const xtl::span<const T>& x, int shape1,
                   std::int64_t shape0_global, std::int64_t rank_offset)
{
  LOG(INFO) << "Send data to post office: ";

  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape1 == 0);
  const std::int64_t shape0 = x.size() / shape1;

  // Post office ranks will receive data from this rank
  std::vector<int> row_to_dest(shape0);
  for (std::int32_t i = 0; i < shape0; ++i)
  {
    int dest = MPI::index_owner(size, i + rank_offset, shape0_global);
    row_to_dest[i] = dest;
  }

  // Remove my rank from array of destination ranks, and then sort
  std::vector<int> dest;
  std::copy_if(row_to_dest.begin(), row_to_dest.end(),
               std::back_insert_iterator(dest),
               [rank](int r) { return r != rank; });
  std::sort(dest.begin(), dest.end());

  // Count number of items (rows of x) to send to each post office (by
  // neighbourhood rank)
  std::vector<std::int32_t> num_items_per_dest;
  {
    auto it = dest.begin();
    while (it != dest.end())
    {
      auto it1 = std::find_if(it, dest.end(),
                              [r = *it](auto idx) { return idx != r; });
      num_items_per_dest.push_back(std::distance(it, it1));
      it = it1;
    }
  }

  // Erase duplicates from array of destination ranks
  dest.erase(std::unique(dest.begin(), dest.end()), dest.end());
  assert(dest.size() == num_items_per_dest.size());

  // Determine source ranks
  const std::vector<int> src = MPI::compute_graph_edges_nbx(comm, dest);
  LOG(INFO) << "Number of neighbourhood source ranks in send_to_postoffice: "
            << src.size();

  // Create neighbourhood communicator for sending data to post offices
  MPI_Comm neigh_comm;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm);

  // Compute send displacements
  std::vector<std::int32_t> send_disp = {0};
  std::partial_sum(num_items_per_dest.begin(), num_items_per_dest.end(),
                   std::back_insert_iterator(send_disp));

  // Pack send buffers
  std::vector<T> send_buffer_data(shape1 * send_disp.back());
  std::vector<std::int64_t> send_buffer_index(send_disp.back());
  {
    std::vector<std::int32_t> send_offsets = send_disp;
    for (std::int32_t i = 0; i < shape0; ++i)
    {
      std::int64_t global_index = i + rank_offset;
      int d = MPI::index_owner(size, global_index, shape0_global);
      if (d != rank)
      {
        // auto neigh_dest_it = global_to_neigh_rank_dest.find(dest);
        auto neigh_dest_it = std::lower_bound(dest.begin(), dest.end(), d);
        assert(neigh_dest_it != dest.end());
        assert(*neigh_dest_it == d);
        int neigh_dest = std::distance(dest.begin(), neigh_dest_it);

        std::size_t offset = send_offsets[neigh_dest];
        send_buffer_index[offset] = global_index;
        for (int j = 0; j < shape1; ++j)
          send_buffer_data[shape1 * offset + j] = x[i * shape1 + j];
        ++send_offsets[neigh_dest];
      }
    }
  }

  // Send number of items to post offices (destination) that I will be
  // sending
  std::vector<int> num_items_recv(src.size());
  MPI_Neighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                        num_items_recv.data(), 1, MPI_INT, neigh_comm);

  // Prepare receive displacement and buffers
  std::vector<std::int32_t> recv_disp = {0};
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::back_insert_iterator(recv_disp));

  MPI_Datatype compound_type;
  MPI_Type_contiguous(shape1, dolfinx::MPI::mpi_type<T>(), &compound_type);
  MPI_Type_commit(&compound_type);

  // Send/receive data (x)
  std::vector<T> recv_buffer_data(shape1 * recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_data.data(), num_items_per_dest.data(),
                         send_disp.data(), compound_type,
                         recv_buffer_data.data(), num_items_recv.data(),
                         recv_disp.data(), compound_type, neigh_comm);

  // Send/receive global indices
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_index.data(), num_items_per_dest.data(),
                         send_disp.data(), MPI_INT64_T,
                         recv_buffer_index.data(), num_items_recv.data(),
                         recv_disp.data(), MPI_INT64_T, neigh_comm);

  MPI_Type_free(&compound_type);
  MPI_Comm_free(&neigh_comm);

  LOG(INFO) << "Completed send data to post offices.";

  return {recv_buffer_index, recv_buffer_data};
};
} // namespace impl

//---------------------------------------------------------------------------
template <typename T>
std::vector<T>
build::distribute_data(MPI_Comm comm,
                       const xtl::span<const std::int64_t>& indices,
                       const xtl::span<const T>& x, int shape1)
{
  common::Timer timer("Fetch data from remote processes");
  assert(shape1 > 0);

  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape1 == 0);
  const std::int64_t shape0 = x.size() / shape1;

  // --------

  std::int64_t shape0_global(0), rank_offset(0);
  MPI_Allreduce(&shape0, &shape0_global, 1, MPI_INT64_T, MPI_SUM, comm);
  MPI_Exscan(&shape0, &rank_offset, 1, MPI_INT64_T, MPI_SUM, comm);

  // Send receive x data to post office (only for rows that need to be
  // moved)
  auto [post_indices, post_x]
      = impl::send_to_postoffice(comm, x, shape1, shape0_global, rank_offset);
  assert(post_indices.size() == post_x.size() / shape1);

  const std::array<std::int64_t, 2> postoffice_range
      = MPI::local_range(rank, shape0_global, size);
  std::vector<std::int32_t> post_indices_new(post_indices.size());
  for (std::size_t i = 0; i < post_indices_new.size(); ++i)
    post_indices_new[i] = post_indices[i] - postoffice_range[0];

  std::vector<std::int32_t> post_indices_map(post_indices_new.size());
  for (std::size_t i = 0; i < post_indices_new.size(); ++i)
    post_indices_map[post_indices_new[i]] = i;

  // Find source post office ranks for my 'indices'
  std::vector<int> index_to_src;
  std::vector<std::pair<int, std::int64_t>> src_to_index;
  for (auto idx : indices)
  {
    int src = MPI::index_owner(size, idx, shape0_global);
    index_to_src.push_back(src);
    if (src != rank)
      src_to_index.push_back({src, idx});
  }

  // Remove my rank from list of source ranks, and then sort
  std::vector<int> src;
  std::copy_if(index_to_src.begin(), index_to_src.end(),
               std::back_insert_iterator(src),
               [rank](int idx) { return idx != rank; });
  std::sort(src.begin(), src.end());

  // Count number of items (rows of x) to receive from each src post
  // office (by neighbourhood rank)
  std::vector<std::int32_t> num_items_per_src;
  {
    auto it = src.begin();
    while (it != src.end())
    {
      auto it1 = std::find_if(it, src.end(),
                              [r = *it](auto idx) { return idx != r; });
      num_items_per_src.push_back(std::distance(it, it1));
      it = it1;
    }
  }

  // Erase duplicates from list of source ranks
  src.erase(std::unique(src.begin(), src.end()), src.end());
  assert(src.size() == num_items_per_src.size());

  // Determine 'delivery' destination ranks (ranks that want data from
  // me)
  const std::vector<int> dest
      = dolfinx::MPI::compute_graph_edges_nbx(comm, src);
  LOG(INFO) << "Neighbourhood destination ranks from post office in "
               "distribute_data (rank, number): "
            << rank << ", " << dest.size();

  // Create neighbourhood communicator for sending data to post offices (src),
  // and receiving data form my send my post office
  MPI_Comm neigh_comm0;
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm0);

  std::vector<int> num_items_recv(dest.size());
  MPI_Neighbor_alltoall(num_items_per_src.data(), 1, MPI_INT,
                        num_items_recv.data(), 1, MPI_INT, neigh_comm0);

  // Prepare send and receive displacements
  std::vector<std::int32_t> send_disp = {0};
  std::partial_sum(num_items_per_src.begin(), num_items_per_src.end(),
                   std::back_insert_iterator(send_disp));
  std::vector<std::int32_t> recv_disp = {0};
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::back_insert_iterator(recv_disp));

  // Pack my requested indices in send buffer ready to send to post
  // offices
  assert(send_disp.back() == (int)src_to_index.size());
  std::vector<std::int64_t> send_buffer_index(send_disp.back() + 1);
  for (std::size_t i = 0; i < src_to_index.size(); ++i)
    send_buffer_index[i] = src_to_index[i].second;

  // Prepare the receive buffer
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_index.data(), num_items_per_src.data(),
                         send_disp.data(), MPI_INT64_T,
                         recv_buffer_index.data(), num_items_recv.data(),
                         recv_disp.data(), MPI_INT64_T, neigh_comm0);

  MPI_Comm_free(&neigh_comm0);

  // Get data and send back (transpose operation)

  // Build send buffer
  std::vector<T> send_buffer_data(shape1 * recv_disp.back());
  for (std::size_t p = 0; p < recv_disp.size() - 1; ++p)
  {
    int offset = recv_disp[p];
    for (std::int32_t i = recv_disp[p]; i < recv_disp[p + 1]; ++i)
    {
      std::int64_t index = recv_buffer_index[i];
      if (index >= rank_offset and index < (rank_offset + shape0))
      {
        // I already had this index before any communication
        std::int32_t local_index = index - rank_offset;
        for (int j = 0; j < shape1; ++j)
          send_buffer_data[shape1 * offset + j] = x[shape1 * local_index + j];
      }
      else
      {
        // Take from my 'post bag'
        auto local_index = index - postoffice_range[0];
        std::int32_t pos = post_indices_map[local_index];
        for (int j = 0; j < shape1; ++j)
          send_buffer_data[shape1 * offset + j] = post_x[shape1 * pos + j];
      }

      ++offset;
    }
  }

  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm0);

  MPI_Datatype compound_type0;
  MPI_Type_contiguous(shape1, dolfinx::MPI::mpi_type<T>(), &compound_type0);
  MPI_Type_commit(&compound_type0);

  std::vector<T> recv_buffer_data(shape1 * send_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_data.data(), num_items_recv.data(),
                         recv_disp.data(), compound_type0,
                         recv_buffer_data.data(), num_items_per_src.data(),
                         send_disp.data(), compound_type0, neigh_comm0);

  MPI_Type_free(&compound_type0);
  MPI_Comm_free(&neigh_comm0);

  std::vector<T> x_new(shape1 * indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    const std::int64_t index = indices[i];
    if (index >= rank_offset and index < (rank_offset + shape0))
    {
      // Had data from the start
      auto local_index = index - rank_offset;
      for (int j = 0; j < shape1; ++j)
        x_new[shape1 * i + j] = x[shape1 * local_index + j];
    }
    else
    {
      if (int src = MPI::index_owner(size, index, shape0_global); src == rank)
      {
        // In my post office bag
        auto local_index = index - postoffice_range[0];
        std::int32_t pos = post_indices_map[local_index];
        for (int j = 0; j < shape1; ++j)
          x_new[shape1 * i + j] = post_x[shape1 * pos + j];
      }
      else
      {
        // In my received post

        // Get index in buffer
        // FIXME: Avoid linear search
        auto it = std::find(send_buffer_index.begin(), send_buffer_index.end(),
                            index);
        assert(it != send_buffer_index.end());
        std::size_t pos = std::distance(send_buffer_index.begin(), it);
        for (int j = 0; j < shape1; ++j)
          x_new[shape1 * i + j] = recv_buffer_data[shape1 * pos + j];
      }
    }
  }

  // --------

  // Get number of rows on each rank
  // const std::int64_t shape0 = x.size() / shape1;
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

  if (x_new == my_x)
    std::cout << "Data matches" << std::endl;
  else
    std::cout << "Data doesn't match" << std::endl;

  return my_x;
}

} // namespace dolfinx::graph
