// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "partition.h"
#include "partitioners.h"
#include <algorithm>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <map>
#include <memory>

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
graph::build::distribute(MPI_Comm comm,
                         const graph::AdjacencyList<std::int64_t>& list,
                         const graph::AdjacencyList<std::int32_t>& destinations)
{
  common::Timer timer("Distribute AdjacencyList nodes to destination ranks");

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
  // Get the maximum number of edges for a node
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

  // Determine source ranks. Sort ranks to make distribution
  // deterministic.
  std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  std::sort(src.begin(), src.end());

  // Create neighbourhood communicator
  MPI_Comm neigh_comm;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm);

  // Send number of nodes to receivers
  std::vector<int> num_items_recv(src.size());
  num_items_per_dest.reserve(1);
  num_items_recv.reserve(1);
  MPI_Request request_size;
  MPI_Ineighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                         num_items_recv.data(), 1, MPI_INT, neigh_comm,
                         &request_size);

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
      const std::size_t pos = dest_data[1];

      std::span b(send_buffer.data() + i * buffer_shape1, buffer_shape1);
      auto row = list.links(pos);
      std::copy(row.begin(), row.end(), b.begin());

      auto info = b.last(3);
      info[0] = row.size();          // Number of edges for node
      info[1] = dest_data[2];        // Owning rank
      info[2] = pos + offset_global; // Original global index
    }
  }

  // Prepare receive displacement
  MPI_Wait(&request_size, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_disp(num_items_recv.size() + 1, 0);
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::next(recv_disp.begin()));

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
  src_ranks0.reserve(recv_disp.back());
  src_ranks1.reserve(recv_disp.back());

  std::vector<std::int64_t> data0, data1;
  data0.reserve((buffer_shape1 - 3) * recv_disp.back());
  data1.reserve((buffer_shape1 - 3) * recv_disp.back());

  std::vector<std::int32_t> offsets0{0}, offsets1{0};
  offsets0.reserve(recv_disp.back());
  offsets1.reserve(recv_disp.back());

  std::vector<std::int64_t> global_indices0, global_indices1;
  global_indices0.reserve(recv_disp.back());
  global_indices1.reserve(recv_disp.back());
  for (std::size_t p = 0; p < recv_disp.size() - 1; ++p)
  {
    const int src_rank = src[p];
    for (std::int32_t i = recv_disp[p]; i < recv_disp[p + 1]; ++i)
    {
      std::span row(recv_buffer.data() + i * buffer_shape1, buffer_shape1);
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
std::vector<std::int64_t>
graph::build::compute_ghost_indices(MPI_Comm comm,
                                    std::span<const std::int64_t> owned_indices,
                                    std::span<const std::int64_t> ghost_indices,
                                    std::span<const int> ghost_owners)
{
  LOG(INFO) << "Compute ghost indices";

  // Get number of local cells determine global offset
  std::int64_t offset_local = 0;
  MPI_Request request_offset_scan;
  const std::int64_t num_local = owned_indices.size();
  MPI_Iexscan(&num_local, &offset_local, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_offset_scan);

  // Find out how many ghosts are on each neighboring process
  std::vector<int> ghost_index_count;
  std::vector<int> neighbors;
  std::map<int, int> proc_to_neighbor;
  {
    int np = 0;
    [[maybe_unused]] int mpi_rank = dolfinx::MPI::rank(comm);
    for (int p : ghost_owners)
    {
      assert(p != mpi_rank);
      auto [it, insert] = proc_to_neighbor.insert({p, np});
      if (insert)
      {
        // New neighbor found
        neighbors.push_back(p);
        ghost_index_count.push_back(0);
        ++np;
      }
      ++ghost_index_count[it->second];
    }
  }

  MPI_Comm neighbor_comm_fwd, neighbor_comm_rev;

  std::vector<int> in_edges = MPI::compute_graph_edges_pcx(comm, neighbors);
  MPI_Dist_graph_create_adjacent(comm, in_edges.size(), in_edges.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbor_comm_fwd);
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, in_edges.size(),
                                 in_edges.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
                                 false, &neighbor_comm_rev);

  std::vector<int> send_offsets = {0};
  send_offsets.reserve(ghost_index_count.size() + 1);
  std::partial_sum(ghost_index_count.begin(), ghost_index_count.end(),
                   std::back_inserter(send_offsets));

  // Copy offsets to help fill array
  std::vector<std::int64_t> send_data(send_offsets.back());
  {
    std::vector<int> ghost_index_offset = send_offsets;
    for (std::size_t i = 0; i < ghost_owners.size(); ++i)
    {
      // Owning process
      int p = ghost_owners[i];

      // Owning neighbor
      int np = proc_to_neighbor[p];

      // Send data location
      int pos = ghost_index_offset[np];
      send_data[pos] = ghost_indices[i];
      ++ghost_index_offset[np];
    }
  }

  std::vector<int> recv_sizes(in_edges.size());
  ghost_index_count.reserve(1);
  recv_sizes.reserve(1);

  MPI_Neighbor_alltoall(ghost_index_count.data(), 1, MPI_INT, recv_sizes.data(),
                        1, MPI_INT, neighbor_comm_fwd);

  std::vector<int> recv_offsets = {0};
  recv_offsets.reserve(recv_sizes.size() + 1);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::back_inserter(recv_offsets));

  std::vector<std::int64_t> recv_data(recv_offsets.back());
  MPI_Neighbor_alltoallv(send_data.data(), ghost_index_count.data(),
                         send_offsets.data(), MPI_INT64_T, recv_data.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT64_T,
                         neighbor_comm_fwd);

  // Complete global_offset scan
  MPI_Wait(&request_offset_scan, MPI_STATUS_IGNORE);

  std::vector<std::array<std::int64_t, 2>> old_to_new;
  old_to_new.reserve(owned_indices.size());

  for (auto idx : owned_indices)
  {
    old_to_new.push_back(
        {idx, static_cast<std::int64_t>(offset_local + old_to_new.size())});
  }
  std::sort(old_to_new.begin(), old_to_new.end());

  // Replace values in recv_data with new_index and send back
  std::transform(recv_data.begin(), recv_data.end(), recv_data.begin(),
                 [&old_to_new](auto r)
                 {
                   auto it = std::lower_bound(
                       old_to_new.begin(), old_to_new.end(),
                       std::array<std::int64_t, 2>{r, 0},
                       [](auto& a, auto& b) { return a[0] < b[0]; });
                   assert(it != old_to_new.end() and (*it)[0] == r);
                   return (*it)[1];
                 });

  std::vector<std::int64_t> new_recv(send_data.size());
  MPI_Neighbor_alltoallv(recv_data.data(), recv_sizes.data(),
                         recv_offsets.data(), MPI_INT64_T, new_recv.data(),
                         ghost_index_count.data(), send_offsets.data(),
                         MPI_INT64_T, neighbor_comm_rev);
  MPI_Comm_free(&neighbor_comm_fwd);
  MPI_Comm_free(&neighbor_comm_rev);

  // Build (old id,  new id) pairs
  std::vector<std::array<std::int64_t, 2>> old_to_new1(send_data.size());
  std::transform(send_data.begin(), send_data.end(), new_recv.begin(),
                 old_to_new1.begin(),
                 [](auto idx_old, auto idx_new) ->
                 typename decltype(old_to_new1)::value_type {
                   return {idx_old, idx_new};
                 });
  std::sort(old_to_new1.begin(), old_to_new1.end());

  std::vector<std::int64_t> ghost_global_indices(ghost_indices.size());
  std::transform(
      ghost_indices.begin(), ghost_indices.end(), ghost_global_indices.begin(),
      [&old_to_new1](auto q)
      {
        auto it
            = std::lower_bound(old_to_new1.begin(), old_to_new1.end(),
                               std::array<std::int64_t, 2>{q, 0},
                               [](auto& a, auto& b) { return a[0] < b[0]; });
        assert(it != old_to_new1.end() and (*it)[0] == q);
        return (*it)[1];
      });

  return ghost_global_indices;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> graph::build::compute_local_to_global_links(
    const graph::AdjacencyList<std::int64_t>& global,
    const graph::AdjacencyList<std::int32_t>& local)
{
  common::Timer timer(
      "Compute-local-to-global links for global/local adjacency list");

  // Return if global and local are empty
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
    std::span<const std::int64_t> local0_to_global,
    std::span<const std::int64_t> local1_to_global)
{
  common::Timer timer("Compute local-to-local map");
  assert(local0_to_global.size() == local1_to_global.size());

  // Compute inverse map for local1_to_global
  std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local1;
  global_to_local1.reserve(local1_to_global.size());
  for (auto idx_global : local1_to_global)
    global_to_local1.push_back({idx_global, global_to_local1.size()});
  std::sort(global_to_local1.begin(), global_to_local1.end());

  // Compute inverse map for local0_to_local1
  std::vector<std::int32_t> local0_to_local1;
  local0_to_local1.reserve(local0_to_global.size());
  std::transform(local0_to_global.begin(), local0_to_global.end(),
                 std::back_inserter(local0_to_local1),
                 [&global_to_local1](auto l2g)
                 {
                   auto it = std::lower_bound(
                       global_to_local1.begin(), global_to_local1.end(),
                       typename decltype(global_to_local1)::value_type(l2g, 0),
                       [](auto& a, auto& b) { return a.first < b.first; });
                   assert(it != global_to_local1.end() and it->first == l2g);
                   return it->second;
                 });

  return local0_to_local1;
}
//-----------------------------------------------------------------------------
