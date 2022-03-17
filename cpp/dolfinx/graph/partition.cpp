// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "partition.h"
#include "partitioners.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <unordered_map>

using namespace dolfinx;

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
graph::partition_graph(MPI_Comm comm, int nparts,
                       const AdjacencyList<std::int64_t>& local_graph,
                       bool ghosting)
{
#if HAS_PARMETIS
  return graph::parmetis::partitioner()(comm, nparts, local_graph, ghosting);
#elif HAS_PTSCOTCH
  return graph::scotch::partitioner()(comm, nparts, local_graph, ghosting);
#elif HAS_KAHIP
  return graph::kahip::partitioner()(comm, nparts, local_graph, ghosting);
#else
// Should never reach this point
#endif
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
graph::build::distribute_new(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& list,
    const graph::AdjacencyList<std::int32_t>& destinations)
{
  common::Timer timer("Distribute AdjacencyList nodes to destination ranks "
                      "(graph::build::distribute, scalable)");

  assert(list.num_nodes() == (int)destinations.num_nodes());
  const int rank = dolfinx::MPI::rank(comm);

  // Get global offset for converting local index to global index for
  // nodes in 'list'
  std::int64_t offset_global = 0;
  {
    const std::int64_t num_owned = list.num_nodes();
    MPI_Exscan(&num_owned, &offset_global, 1, MPI_INT64_T, MPI_SUM, comm);
  }

  // TODO: Do this on the neighbourhood only
  // Get the maximum number of eddges for a node
  int shape1 = 0;
  {
    int shape1_local = list.num_nodes() > 0 ? list.links(0).size() : 0;
    MPI_Allreduce(&shape1_local, &shape1, 1, MPI_INT, MPI_MAX, comm);
  }

  // Buffer size (max number of edges + 3 for num_edges, owning rank,
  // and node global index)
  const std::size_t buffer_shape1 = shape1 + 3;

  // Build (dest, index, owning rank) list and sort
  std::vector<std::array<int, 3>> dest_to_index;
  dest_to_index.reserve(destinations.array().size());
  for (std::int32_t i = 0; i < destinations.num_nodes(); ++i)
  {
    auto di = destinations.links(i);
    for (auto d : di)
      dest_to_index.push_back({d, i, di[0]});
  }
  std::sort(dest_to_index.begin(), dest_to_index.end());

  // Build list of unique dest ranks and count number of rows to send to
  // each dest (by neighbourhood rank)
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest;
  {
    auto it = dest_to_index.begin();
    while (it != dest_to_index.end())
    {
      // Store global rank and find iterator to next global rank
      dest.push_back((*it)[0]);
      auto it1
          = std::find_if(it, dest_to_index.end(),
                         [r = dest.back()](auto& idx) { return idx[0] != r; });

      // Store number of items for current rank
      num_items_per_dest.push_back(std::distance(it, it1));

      // Advance iterator
      it = it1;
    }
  }

  // Determine source ranks
  const std::vector<int> src
      = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);

  // Create neighbourhood communicator
  MPI_Comm neigh_comm;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm);

  // Send number of nodes to receivers
  std::vector<int> num_items_recv(src.size());
  num_items_per_dest.reserve(1);
  num_items_recv.reserve(1);
  MPI_Neighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                        num_items_recv.data(), 1, MPI_INT, neigh_comm);

  // Prepare receive displacement
  std::vector<std::int32_t> recv_disp(num_items_recv.size() + 1, 0);
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::next(recv_disp.begin()));

  // Compute send displacements
  std::vector<std::int32_t> send_disp(num_items_per_dest.size() + 1, 0);
  std::partial_sum(num_items_per_dest.begin(), num_items_per_dest.end(),
                   std::next(send_disp.begin()));

  // Pack send buffer
  std::vector<std::int64_t> send_buffer(buffer_shape1 * send_disp.back(), -1);
  {
    assert(send_disp.back() == (std::int32_t)dest_to_index.size());
    for (std::size_t i = 0; i < dest_to_index.size(); ++i)
    {
      const std::array<int, 3>& dest_data = dest_to_index[i];
      xtl::span b(send_buffer.data() + i * buffer_shape1, buffer_shape1);
      auto row = list.links(dest_data[1]);
      std::copy(row.begin(), row.end(), b.begin());

      auto info = b.last(3);
      info[0] = row.size();          // Number of edges for node
      info[1] = dest_data[2];        // Owning rank
      info[2] = pos + offset_global; // Original global index
    }
  }

  // Send/receive data facet
  MPI_Datatype compound_type;
  MPI_Type_contiguous(buffer_shape1, MPI_INT64_T, &compound_type);
  MPI_Type_commit(&compound_type);
  std::vector<std::int64_t> recv_buffer(buffer_shape1 * recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer.data(), num_items_per_dest.data(),
                         send_disp.data(), compound_type, recv_buffer.data(),
                         num_items_recv.data(), recv_disp.data(), compound_type,
                         neigh_comm);

  MPI_Type_free(&compound_type);
  MPI_Comm_free(&neigh_comm);

  // Unpack receive buffer
  std::vector<int> src_ranks0, src_ranks1, ghost_index_owner;
  std::vector<std::int64_t> data0, data1;
  std::vector<std::int32_t> offsets0{0}, offsets1{0};
  std::vector<std::int64_t> global_indices0, global_indices1;
  for (std::size_t p = 0; p < recv_disp.size() - 1; ++p)
  {
    const int src_rank = src[p];
    for (std::int32_t i = recv_disp[p]; i < recv_disp[p + 1]; ++i)
    {
      xtl::span row(recv_buffer.data() + i * buffer_shape1, buffer_shape1);
      auto info = row.last(3);
      std::size_t num_edges = info[0];
      int owner = info[1];
      std::int64_t orig_global_index = info[2];
      auto edges = row.first(num_edges);
      if (owner == rank)
      {
        data0.insert(data0.end(), edges.begin(), edges.end());
        offsets0.push_back(offsets0.back() + num_edges);
        src_ranks0.push_back(src_rank);
        global_indices0.push_back(orig_global_index);
      }
      else
      {
        data1.insert(data1.end(), edges.begin(), edges.end());
        offsets1.push_back(offsets1.back() + info[0]);
        src_ranks1.push_back(src_rank);
        global_indices1.push_back(orig_global_index);

        ghost_index_owner.push_back(info[1]);
      }
    }
  }

  std::transform(offsets1.begin(), offsets1.end(), offsets1.begin(),
                 [off = offsets0.back()](auto x) { return x + off; });
  data0.insert(data0.end(), data1.begin(), data1.end());
  offsets0.insert(offsets0.end(), std::next(offsets1.begin()), offsets1.end());
  src_ranks0.insert(src_ranks0.end(), src_ranks1.begin(), src_ranks1.end());
  global_indices0.insert(global_indices0.end(), global_indices1.begin(),
                         global_indices1.end());

  data0.shrink_to_fit();
  offsets0.shrink_to_fit();
  src_ranks0.shrink_to_fit();
  global_indices0.shrink_to_fit();
  ghost_index_owner.shrink_to_fit();

  return {graph::AdjacencyList<std::int64_t>(data0, offsets0), src_ranks0,
          global_indices0, ghost_index_owner};
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
graph::build::distribute(MPI_Comm comm,
                         const graph::AdjacencyList<std::int64_t>& list,
                         const graph::AdjacencyList<std::int32_t>& destinations)
{
  common::Timer timer("Distribute AdjacencyList nodes to destination ranks "
                      "(graph::build::distribute)");

  assert(list.num_nodes() == (int)destinations.num_nodes());

  std::int64_t offset_global = 0;
  const std::int64_t num_owned = list.num_nodes();
  MPI_Request request_offset_scan;
  MPI_Iexscan(&num_owned, &offset_global, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_offset_scan);

  const int size = dolfinx::MPI::size(comm);

  // Compute number of links to send to each process
  std::vector<int> num_per_dest_send(size, 0);
  for (int i = 0; i < destinations.num_nodes(); ++i)
  {
    int list_num_links = list.num_links(i) + 3;
    auto dests = destinations.links(i);
    for (std::int32_t d : dests)
      num_per_dest_send[d] += list_num_links;
  }

  // Compute send array displacements
  std::vector<int> disp_send(size + 1, 0);
  std::partial_sum(num_per_dest_send.begin(), num_per_dest_send.end(),
                   disp_send.begin() + 1);

  // Send/receive number of items to communicate
  std::vector<int> num_per_dest_recv(size, 0);
  MPI_Alltoall(num_per_dest_send.data(), 1, MPI_INT, num_per_dest_recv.data(),
               1, MPI_INT, comm);

  // Compute receive array displacements
  std::vector<int> disp_recv(size + 1, 0);
  std::partial_sum(num_per_dest_recv.begin(), num_per_dest_recv.end(),
                   disp_recv.begin() + 1);

  // Complete global_offset scan
  MPI_Wait(&request_offset_scan, MPI_STATUS_IGNORE);

  // Prepare send buffer
  std::vector<int> offset = disp_send;
  std::vector<std::int64_t> data_send(disp_send.back());
  for (int i = 0; i < list.num_nodes(); ++i)
  {
    auto links = list.links(i);
    auto dests = destinations.links(i);
    for (auto dest : dests)
    {
      data_send[offset[dest]++] = dests[0];
      data_send[offset[dest]++] = i + offset_global;
      data_send[offset[dest]++] = links.size();
      std::copy(links.cbegin(), links.cend(),
                std::next(data_send.begin(), offset[dest]));
      offset[dest] += links.size();
    }
  }

  // Send/receive data
  std::vector<std::int64_t> data_recv(disp_recv.back());
  MPI_Alltoallv(data_send.data(), num_per_dest_send.data(), disp_send.data(),
                MPI_INT64_T, data_recv.data(), num_per_dest_recv.data(),
                disp_recv.data(), MPI_INT64_T, comm);

  // Force memory to be freed
  std::vector<int>().swap(num_per_dest_send);
  std::vector<int>().swap(disp_send);
  std::vector<int>().swap(num_per_dest_recv);
  std::vector<std::int64_t>().swap(data_send);

  // Unpack receive buffer
  int mpi_rank = dolfinx::MPI::rank(comm);
  std::vector<std::int64_t> array;
  std::vector<std::int64_t> ghost_array;
  std::vector<std::int64_t> global_indices;
  std::vector<std::int64_t> ghost_global_indices;
  std::vector<std::int32_t> list_offset = {0};
  std::vector<std::int32_t> ghost_list_offset = {0};
  std::vector<int> src;
  std::vector<int> ghost_src;
  std::vector<int> ghost_index_owner;

  for (std::size_t p = 0; p < disp_recv.size() - 1; ++p)
  {
    for (int i = disp_recv[p]; i < disp_recv[p + 1];)
    {
      if (data_recv[i] == mpi_rank)
      {
        src.push_back(p);
        i++; // index_owner.push_back(data_recv[i++]);
        global_indices.push_back(data_recv[i++]);
        const std::int64_t num_links = data_recv[i++];
        array.insert(array.end(), std::next(data_recv.begin(), i),
                     std::next(data_recv.begin(), i + num_links));
        i += num_links;
        list_offset.push_back(list_offset.back() + num_links);
      }
      else
      {
        ghost_src.push_back(p);
        ghost_index_owner.push_back(data_recv[i++]);
        ghost_global_indices.push_back(data_recv[i++]);
        const std::int64_t num_links = data_recv[i++];
        ghost_array.insert(ghost_array.end(), std::next(data_recv.begin(), i),
                           std::next(data_recv.begin(), i + num_links));
        i += num_links;
        ghost_list_offset.push_back(ghost_list_offset.back() + num_links);
      }
    }
  }

  // Attach all ghost cells at the end of the list
  src.insert(src.end(), ghost_src.begin(), ghost_src.end());
  global_indices.insert(global_indices.end(), ghost_global_indices.begin(),
                        ghost_global_indices.end());
  array.insert(array.end(), ghost_array.begin(), ghost_array.end());
  int ghost_offset = list_offset.back();
  list_offset.pop_back();
  std::for_each(ghost_list_offset.begin(), ghost_list_offset.end(),
                [ghost_offset](auto& offset) { offset += ghost_offset; });
  list_offset.insert(list_offset.end(), ghost_list_offset.begin(),
                     ghost_list_offset.end());

  array.shrink_to_fit();
  list_offset.shrink_to_fit();
  src.shrink_to_fit();
  global_indices.shrink_to_fit();
  ghost_index_owner.shrink_to_fit();

  // auto [g, xsrc, xglobal_indices, xghost_index_owner]
  //     = distribute_new(comm, list, destinations);

  return distribute_new(comm, list, destinations);
  // if (dolfinx::MPI::rank(comm) == 1)
  // {
  //   auto tmp = graph::AdjacencyList<std::int64_t>(array, list_offset);
  //   std::cout << tmp.str() << std::endl;
  //   std::cout << "src" << std::endl;
  //   for (auto x : src)
  //     std::cout << "   " << x << std::endl;
  //   std::cout << "global_indices" << std::endl;
  //   for (auto x : global_indices)
  //     std::cout << "   " << x << std::endl;
  //   std::cout << "ghost_index_owner" << std::endl;
  //   for (auto x : ghost_index_owner)
  //     std::cout << "   " << x << std::endl;

  //   std::cout << g.str() << std::endl;
  //   std::cout << "src" << std::endl;
  //   for (auto x : xsrc)
  //     std::cout << "   " << x << std::endl;
  //   std::cout << "global_indices" << std::endl;
  //   for (auto x : xglobal_indices)
  //     std::cout << "   " << x << std::endl;
  //   std::cout << "ghost_index_owner" << std::endl;
  //   for (auto x : xghost_index_owner)
  //     std::cout << "   " << x << std::endl;
  // }

  // if (g.array() != array)
  //   std::cout << "Array mis-match " << dolfinx::MPI::rank(comm) << std::endl;
  // if (g.offsets() != list_offset)
  //   std::cout << "Offset mis-match " << dolfinx::MPI::rank(comm) <<
  //   std::endl;
  // if (src != xsrc)
  //   std::cout << "src mis-match " << dolfinx::MPI::rank(comm) << std::endl;
  // if (global_indices != xglobal_indices)
  //   std::cout << "global_indices mis-match " << dolfinx::MPI::rank(comm)
  //             << std::endl;
  // if (ghost_index_owner != xghost_index_owner)
  //   std::cout << "ghost_index_owner mis-match " << dolfinx::MPI::rank(comm)
  //             << std::endl;

  // return {graph::AdjacencyList<std::int64_t>(std::move(array),
  //                                            std::move(list_offset)),
  //         std::move(src), std::move(global_indices),
  //         std::move(ghost_index_owner)};
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> graph::build::compute_ghost_indices(
    MPI_Comm comm, const xtl::span<const std::int64_t>& global_indices,
    const xtl::span<const int>& ghost_owners)
{
  LOG(INFO) << "Compute ghost indices";

  // Get number of local cells and global offset
  const std::int64_t num_local = global_indices.size() - ghost_owners.size();
  std::vector<std::int64_t> ghost_global_indices(
      global_indices.begin() + num_local, global_indices.end());

  std::int64_t offset_local = 0;
  MPI_Request request_offset_scan;
  MPI_Iexscan(&num_local, &offset_local, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_offset_scan);

  // Find out how many ghosts are on each neighboring process
  std::vector<int> ghost_index_count;
  std::vector<int> neighbors;
  std::map<int, int> proc_to_neighbor;
  int np = 0;
  [[maybe_unused]] int mpi_rank = dolfinx::MPI::rank(comm);
  for (int p : ghost_owners)
  {
    assert(p != mpi_rank);
    const auto [it, insert] = proc_to_neighbor.insert({p, np});
    if (insert)
    {
      // New neighbor found
      neighbors.push_back(p);
      ghost_index_count.push_back(0);
      ++np;
    }
    ++ghost_index_count[it->second];
  }

  // NB - this assumes a symmetry, i.e. that if one process shares an index
  // owned by another process, then the same is true vice versa. This
  // assumption is valid for meshes with cells shared via facet or vertex.
  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbor_comm);

  std::vector<int> send_offsets = {0};
  send_offsets.reserve(ghost_index_count.size() + 1);
  for (int index_count : ghost_index_count)
    send_offsets.push_back(send_offsets.back() + index_count);
  std::vector<std::int64_t> send_data(send_offsets.back());

  // Copy offsets to help fill array
  std::vector<int> ghost_index_offset(send_offsets.begin(), send_offsets.end());

  for (std::size_t i = 0; i < ghost_owners.size(); ++i)
  {
    // Owning process
    int p = ghost_owners[i];

    // Owning neighbor
    int np = proc_to_neighbor[p];

    // Send data location
    int pos = ghost_index_offset[np];
    send_data[pos] = global_indices[num_local + i];
    ++ghost_index_offset[np];
  }

  std::vector<int> recv_sizes(neighbors.size());
  ghost_index_count.reserve(1);
  recv_sizes.reserve(1);
  MPI_Neighbor_alltoall(ghost_index_count.data(), 1, MPI_INT, recv_sizes.data(),
                        1, MPI_INT, neighbor_comm);
  std::vector<int> recv_offsets = {0};
  recv_offsets.reserve(recv_sizes.size() + 1);
  for (int q : recv_sizes)
    recv_offsets.push_back(recv_offsets.back() + q);

  std::vector<std::int64_t> recv_data(recv_offsets.back());
  MPI_Neighbor_alltoallv(send_data.data(), ghost_index_count.data(),
                         send_offsets.data(), MPI_INT64_T, recv_data.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT64_T,
                         neighbor_comm);

  // Complete global_offset scan
  MPI_Wait(&request_offset_scan, MPI_STATUS_IGNORE);

  // Replace values in recv_data with new_index and send back
  std::unordered_map<std::int64_t, std::int64_t> old_to_new;
  for (int i = 0; i < num_local; ++i)
    old_to_new.insert({global_indices[i], offset_local + i});

  std::for_each(recv_data.begin(), recv_data.end(),
                [&old_to_new](auto& r)
                {
                  auto it = old_to_new.find(r);
                  // Must exist on this process!
                  assert(it != old_to_new.end());
                  r = it->second;
                });

  std::vector<std::int64_t> new_recv(send_data.size());
  MPI_Neighbor_alltoallv(recv_data.data(), recv_sizes.data(),
                         recv_offsets.data(), MPI_INT64_T, new_recv.data(),
                         ghost_index_count.data(), send_offsets.data(),
                         MPI_INT64_T, neighbor_comm);

  // Add to map
  for (std::size_t i = 0; i < send_data.size(); ++i)
  {
    std::int64_t old_idx = send_data[i];
    std::int64_t new_idx = new_recv[i];
    [[maybe_unused]] auto [it, insert] = old_to_new.insert({old_idx, new_idx});
    assert(insert);
  }

  std::for_each(ghost_global_indices.begin(), ghost_global_indices.end(),
                [&old_to_new](auto& q)
                {
                  const auto it = old_to_new.find(q);
                  assert(it != old_to_new.end());
                  q = it->second;
                });

  MPI_Comm_free(&neighbor_comm);
  return ghost_global_indices;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> graph::build::compute_local_to_global_links(
    const graph::AdjacencyList<std::int64_t>& global,
    const graph::AdjacencyList<std::int32_t>& local)
{
  common::Timer timer(
      "Compute-local-to-global links for global/local adjacency list");

  // Return if gloabl and local are empty
  if (global.num_nodes() == 0 and local.num_nodes() == 0)
    return std::vector<std::int64_t>();

  // Build local-to-global for adjacency lists
  if (global.num_nodes() != local.num_nodes())
  {
    throw std::runtime_error("Mismatch in number of nodes between local and "
                             "global adjacency lists.");
  }

  const std::vector<std::int64_t>& _global = global.array();
  const std::vector<std::int32_t>& _local = local.array();
  if (_global.size() != _local.size())
  {
    throw std::runtime_error("Data size mismatch between local and "
                             "global adjacency lists.");
  }

  const std::int32_t max_local_idx
      = *std::max_element(_local.begin(), _local.end());
  std::vector<std::int64_t> local_to_global_list(max_local_idx + 1, -1);
  for (std::size_t i = 0; i < _local.size(); ++i)
  {
    if (local_to_global_list[_local[i]] == -1)
      local_to_global_list[_local[i]] = _global[i];
  }

  return local_to_global_list;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> graph::build::compute_local_to_local(
    const xtl::span<const std::int64_t>& local0_to_global,
    const xtl::span<const std::int64_t>& local1_to_global)
{
  common::Timer timer("Compute local-to-local map");
  assert(local0_to_global.size() == local1_to_global.size());

  // Compute inverse map for local1_to_global
  std::unordered_map<std::int64_t, std::int32_t> global_to_local1;
  for (std::size_t i = 0; i < local1_to_global.size(); ++i)
    global_to_local1.insert({local1_to_global[i], i});

  // Compute inverse map for local0_to_local1
  std::vector<std::int32_t> local0_to_local1;
  std::transform(local0_to_global.cbegin(), local0_to_global.cend(),
                 std::back_inserter(local0_to_local1),
                 [&global_to_local1](auto l2g)
                 {
                   auto it = global_to_local1.find(l2g);
                   assert(it != global_to_local1.end());
                   return it->second;
                 });

  return local0_to_local1;
}
//-----------------------------------------------------------------------------
