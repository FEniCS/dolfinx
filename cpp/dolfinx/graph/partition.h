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

/// @brief Distribute rows of a rectangular data array to process ranks
/// where they are required.
///
/// @param[in] comm The MPI communicator
/// @param[in] indices Global indices of the data (rows) required by
/// this process
/// @param[in] x Data on this process which may be distributed (by row).
/// The global index for the `[0, ..., n)` local rows is assumed to be
/// the local index plus the offset for this rank. Layout is row-major.
/// @param[in] shape1 The number of columns of the data array
/// @return The data for each index in `indices` (row-major storage)
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
std::pair<std::vector<std::int32_t>, std::vector<T>>
send_to_postoffice(MPI_Comm comm, const xtl::span<const T>& x, int shape1,
                   std::int64_t shape0_global, std::int64_t rank_offset)
{
  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape1 == 0);
  const std::int32_t shape0 = x.size() / shape1;

  LOG(2) << "Sending data to post offices (send_to_postoffice)";

  // Post office ranks will receive data from this rank
  std::vector<int> row_to_dest(shape0);
  for (std::int32_t i = 0; i < shape0; ++i)
  {
    int dest = MPI::index_owner(size, i + rank_offset, shape0_global);
    row_to_dest[i] = dest;
  }

  // Build list of (dest, positions) for each row that doesn't belong to
  // this rank, then sort
  std::vector<std::array<std::int32_t, 2>> dest_to_index;
  dest_to_index.reserve(shape0);
  for (std::int32_t i = 0; i < shape0; ++i)
  {
    std::size_t idx = i + rank_offset;
    if (int dest = MPI::index_owner(size, idx, shape0_global); dest != rank)
      dest_to_index.push_back({dest, i});
  }
  std::sort(dest_to_index.begin(), dest_to_index.end());

  // Build list is neighbour src ranks and count number of items (rows
  // of x) to receive from each src post office (by neighbourhood rank)
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest, pos_to_neigh_rank(shape0, -1);
  {
    auto it = dest_to_index.begin();
    while (it != dest_to_index.end())
    {
      const int neigh_rank = dest.size();

      // Store global rank
      dest.push_back((*it)[0]);

      // Find iterator to next global rank
      auto it1
          = std::find_if(it, dest_to_index.end(),
                         [r = dest.back()](auto& idx) { return idx[0] != r; });

      // Store number of items for current rank
      num_items_per_dest.push_back(std::distance(it, it1));

      // Map from local x index to local destination rank
      for (auto e = it; e != it1; ++e)
        pos_to_neigh_rank[(*e)[1]] = neigh_rank;

      // Advance iterator
      it = it1;
    }
  }

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
      if (int neigh_dest = pos_to_neigh_rank[i]; neigh_dest != -1)
      {
        std::size_t pos = send_offsets[neigh_dest];
        send_buffer_index[pos] = i + rank_offset;
        std::copy_n(std::next(x.begin(), i * shape1), shape1,
                    std::next(send_buffer_data.begin(), shape1 * pos));
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

  // Send/receive global indices
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_index.data(), num_items_per_dest.data(),
                         send_disp.data(), MPI_INT64_T,
                         recv_buffer_index.data(), num_items_recv.data(),
                         recv_disp.data(), MPI_INT64_T, neigh_comm);

  // Send/receive data (x)
  MPI_Datatype compound_type;
  MPI_Type_contiguous(shape1, dolfinx::MPI::mpi_type<T>(), &compound_type);
  MPI_Type_commit(&compound_type);
  std::vector<T> recv_buffer_data(shape1 * recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_data.data(), num_items_per_dest.data(),
                         send_disp.data(), compound_type,
                         recv_buffer_data.data(), num_items_recv.data(),
                         recv_disp.data(), compound_type, neigh_comm);
  MPI_Type_free(&compound_type);

  MPI_Comm_free(&neigh_comm);

  LOG(2) << "Completed send data to post offices.";

  // Convert to local indices
  const std::int64_t r0 = MPI::local_range(rank, shape0_global, size)[0];
  std::vector<std::int32_t> index_local(recv_buffer_index.size());
  std::transform(recv_buffer_index.cbegin(), recv_buffer_index.cend(),
                 index_local.begin(), [r0](auto idx) { return idx - r0; });

  return {index_local, recv_buffer_data};
};
} // namespace impl

//---------------------------------------------------------------------------
template <typename T>
std::vector<T>
build::distribute_data(MPI_Comm comm,
                       const xtl::span<const std::int64_t>& indices,
                       const xtl::span<const T>& x, int shape1)
{
  common::Timer timer("Fetch row-wise data from remote processes");
  assert(shape1 > 0);

  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape1 == 0);
  const std::int64_t shape0 = x.size() / shape1;

  // --------

  std::int64_t shape0_global(0), rank_offset(0);
  MPI_Allreduce(&shape0, &shape0_global, 1, MPI_INT64_T, MPI_SUM, comm);
  MPI_Exscan(&shape0, &rank_offset, 1, MPI_INT64_T, MPI_SUM, comm);

  // 0. Send x data to/from post offices

  // Send receive x data to post office (only for rows that need to be
  // communicated)
  auto [post_indices, post_x]
      = impl::send_to_postoffice(comm, x, shape1, shape0_global, rank_offset);
  assert(post_indices.size() == post_x.size() / shape1);

  // 1. Send request to post office ranks for data

  // Build list of (src, global index, global, index positions) for each
  // entry in 'indices' that doesn't belong to this rank, then sort
  std::vector<std::tuple<int, std::int64_t, std::int32_t>> src_to_index;
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    std::size_t idx = indices[i];
    if (int src = MPI::index_owner(size, idx, shape0_global); src != rank)
      src_to_index.push_back({src, idx, i});
  }
  std::sort(src_to_index.begin(), src_to_index.end());

  // Build list is neighbour src ranks and count number of items (rows
  // of x) to receive from each src post office (by neighbourhood rank)
  std::vector<std::int32_t> num_items_per_src;
  std::vector<int> src;
  {
    auto it = src_to_index.begin();
    while (it != src_to_index.end())
    {
      src.push_back(std::get<0>(*it));
      auto it1 = std::find_if(it, src_to_index.end(),
                              [r = src.back()](auto& idx)
                              { return std::get<0>(idx) != r; });
      num_items_per_src.push_back(std::distance(it, it1));
      it = it1;
    }
  }

  // Determine 'delivery' destination ranks (ranks that want data from
  // me)
  const std::vector<int> dest
      = dolfinx::MPI::compute_graph_edges_nbx(comm, src);
  LOG(INFO) << "Neighbourhood destination ranks from post office in "
               "distribute_data (rank, num dests, num dests/mpi_size): "
            << rank << ", " << dest.size() << ", "
            << static_cast<double>(dest.size()) / size;

  // Create neighbourhood communicator for sending data to post offices
  // (src), and receiving data form my send my post office
  MPI_Comm neigh_comm0;
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm0);

  // Communicate number of requests to each source
  std::vector<int> num_items_recv(dest.size());
  MPI_Neighbor_alltoall(num_items_per_src.data(), 1, MPI_INT,
                        num_items_recv.data(), 1, MPI_INT, neigh_comm0);

  // Prepare send/receive displacements
  std::vector<std::int32_t> send_disp = {0};
  std::partial_sum(num_items_per_src.begin(), num_items_per_src.end(),
                   std::back_insert_iterator(send_disp));
  std::vector<std::int32_t> recv_disp = {0};
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::back_insert_iterator(recv_disp));

  // Pack my requested indices (global) in send buffer ready to send to
  // post offices
  assert(send_disp.back() == (int)src_to_index.size());
  std::vector<std::int64_t> send_buffer_index(src_to_index.size());
  std::transform(src_to_index.cbegin(), src_to_index.cend(),
                 send_buffer_index.begin(),
                 [](auto& x) { return std::get<1>(x); });

  // Prepare the receive buffer
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_index.data(), num_items_per_src.data(),
                         send_disp.data(), MPI_INT64_T,
                         recv_buffer_index.data(), num_items_recv.data(),
                         recv_disp.data(), MPI_INT64_T, neigh_comm0);

  MPI_Comm_free(&neigh_comm0);

  // 2. Send data (rows of x) back to requesting ranks (transpose of the
  // preceding communication pattern operation)

  // Build map from local index to post_indices position. Set to -1 for
  // data that was already on this rank and was therefore was not
  // sent/received via a postoffice.
  const std::array<std::int64_t, 2> postoffice_range
      = MPI::local_range(rank, shape0_global, size);
  std::vector<std::int32_t> post_indices_map(
      postoffice_range[1] - postoffice_range[0], -1);
  for (std::size_t i = 0; i < post_indices.size(); ++i)
  {
    assert(post_indices[i] < (int)post_indices_map.size());
    post_indices_map[post_indices[i]] = i;
  }

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
        std::copy_n(std::next(x.begin(), shape1 * local_index), shape1,
                    std::next(send_buffer_data.begin(), shape1 * offset));
      }
      else
      {
        // Take from my 'post bag'
        auto local_index = index - postoffice_range[0];
        std::int32_t pos = post_indices_map[local_index];
        assert(pos != -1);
        std::copy_n(std::next(post_x.begin(), shape1 * pos), shape1,
                    std::next(send_buffer_data.begin(), shape1 * offset));
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

  std::vector<std::int32_t> index_pos_to_buffer(indices.size(), -1);
  for (std::size_t i = 0; i < src_to_index.size(); ++i)
    index_pos_to_buffer[std::get<2>(src_to_index[i])] = i;

  // Extra data to return
  std::vector<T> x_new(shape1 * indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    const std::int64_t index = indices[i];
    if (index >= rank_offset and index < (rank_offset + shape0))
    {
      // Had data from the start in x
      auto local_index = index - rank_offset;
      std::copy_n(std::next(x.begin(), shape1 * local_index), shape1,
                  std::next(x_new.begin(), shape1 * i));
    }
    else
    {
      if (int src = MPI::index_owner(size, index, shape0_global); src == rank)
      {
        // In my post office bag
        auto local_index = index - postoffice_range[0];
        std::int32_t pos = post_indices_map[local_index];
        assert(pos != -1);
        std::copy_n(std::next(post_x.begin(), shape1 * pos), shape1,
                    std::next(x_new.begin(), shape1 * i));
      }
      else
      {
        // In my received post
        std::int32_t pos = index_pos_to_buffer[i];
        assert(pos != -1);
        std::copy_n(std::next(recv_buffer_data.begin(), shape1 * pos), shape1,
                    std::next(x_new.begin(), shape1 * i));
      }
    }
  }
  timer.stop();

  // --------
  // Original code

  common::Timer timer1("Fetch row-wise data from remote processes (OLD)");

  // Get number of rows on each rank
  std::vector<std::int64_t> global_sizes(size);
  MPI_Allgather(&shape0, 1, MPI_INT64_T, global_sizes.data(), 1, MPI_INT64_T,
                comm);
  std::vector<std::int64_t> global_offsets(size + 1, 0);
  std::partial_sum(global_sizes.begin(), global_sizes.end(),
                   std::next(global_offsets.begin()));

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
                   std::next(disp_index_send.begin()));

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
                   std::next(disp_index_recv.begin()));

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

  timer1.stop();

  if (x_new != my_x)
    throw std::runtime_error("Parallel data does not match");
  // if (x_new == my_x)
  //   std::cout << "*** Data matches: " << rank << ", " << x_new.size() << ", "
  //             << my_x.size() << std::endl;
  // else
  //   std::cout << "*** Data doesn't match: " << rank << ", " << x_new.size()
  //             << ", " << my_x.size() << std::endl;

  return my_x;
}

} // namespace dolfinx::graph
