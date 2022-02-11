// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "partition.h"
#include "partitioners.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <unordered_map>

using namespace dolfinx;

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
graph::partition_graph(MPI_Comm comm, int nparts,
                       const AdjacencyList<std::int64_t>& local_graph,
                       std::int32_t num_ghost_nodes, bool ghosting)
{
#if HAS_PARMETIS
  return graph::parmetis::partitioner()(comm, nparts, local_graph,
                                        num_ghost_nodes, ghosting);
#elif HAS_PTSCOTCH
  return graph::scotch::partitioner()(comm, nparts, local_graph,
                                      num_ghost_nodes, ghosting);
#elif HAS_KAHIP
  return graph::kahip::partitioner()(comm, nparts, local_graph, num_ghost_nodes,
                                     ghosting);
#else
// Should never reach this point
#endif
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
graph::build::distribute(MPI_Comm comm,
                         const graph::AdjacencyList<std::int64_t>& list,
                         const graph::AdjacencyList<std::int32_t>& destinations)
{
  common::Timer timer("Distribute in graph creation AdjacencyList");

  assert(list.num_nodes() == (int)destinations.num_nodes());

  std::int64_t offset_global = 0;
  const std::int64_t num_owned = list.num_nodes();
  MPI_Request request_offset_scan;
  MPI_Iexscan(&num_owned, &offset_global, 1,
              dolfinx::MPI::mpi_type<std::int64_t>(), MPI_SUM, comm,
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
  return {graph::AdjacencyList<std::int64_t>(std::move(array),
                                             std::move(list_offset)),
          std::move(src), std::move(global_indices),
          std::move(ghost_index_owner)};
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
  MPI_Iexscan(&num_local, &offset_local, 1,
              dolfinx::MPI::mpi_type<std::int64_t>(), MPI_SUM, comm,
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

  std::vector<int> recv_sizes(neighbors.size() + 1);
  ghost_index_count.push_back(0);
  MPI_Neighbor_alltoall(ghost_index_count.data(), 1, MPI_INT, recv_sizes.data(),
                        1, MPI_INT, neighbor_comm);
  ghost_index_count.pop_back();
  recv_sizes.pop_back();
  std::vector<int> recv_offsets = {0};
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
