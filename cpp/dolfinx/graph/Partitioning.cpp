// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Partitioning.h"
#include <Eigen/Dense>
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/SCOTCH.h>
#include <unordered_map>

using namespace dolfinx;
using namespace dolfinx::graph;

//-----------------------------------------------------------------------------
std::tuple<std::vector<std::int32_t>, std::vector<std::int64_t>,
           std::vector<int>>
Partitioning::reorder_global_indices(
    MPI_Comm comm, const std::vector<std::int64_t>& global_indices,
    const std::vector<bool>& shared_indices)
{
  common::Timer timer("Re-order global indices");

  // TODO: Can this function be broken into multiple logical steps?

  assert(global_indices.size() == shared_indices.size());

  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);

  // Get maximum global index across all processes
  auto it_max = std::max_element(global_indices.begin(), global_indices.end());
  const std::int64_t my_max_global_index
      = (it_max != global_indices.end()) ? *it_max : 0;
  std::int64_t max_global_index = 0;
  MPI_Allreduce(&my_max_global_index, &max_global_index, 1, MPI_INT64_T,
                MPI_MAX, comm);

  // Create global ->local map
  std::unordered_map<std::int64_t, std::int32_t> global_to_local;
  for (std::size_t i = 0; i < global_indices.size(); ++i)
    global_to_local.insert({global_indices[i], i});

  // Compute number of possibly shared vertices to send to each process,
  // considering only vertices that are possibly shared
  std::vector<int> number_send(size, 0);
  for (auto& vertex : global_to_local)
  {
    if (shared_indices[vertex.second])
    {
      const int owner
          = dolfinx::MPI::index_owner(size, vertex.first, max_global_index + 1);
      number_send[owner] += 1;
    }
  }

  // Compute send displacements
  std::vector<int> disp_send(size + 1, 0);
  std::partial_sum(number_send.begin(), number_send.end(),
                   disp_send.begin() + 1);

  // Pack global index send data
  std::vector<std::int64_t> indices_send(disp_send.back());
  std::vector<int> disp_tmp = disp_send;
  for (auto vertex : global_to_local)
  {
    if (shared_indices[vertex.second])
    {
      const int owner
          = dolfinx::MPI::index_owner(size, vertex.first, max_global_index + 1);
      indices_send[disp_tmp[owner]++] = vertex.first;
    }
  }

  // Send/receive number of indices to communicate to each process
  std::vector<int> number_recv(size);
  MPI_Alltoall(number_send.data(), 1, MPI_INT, number_recv.data(), 1, MPI_INT,
               comm);

  // Compute receive displacements
  std::vector<int> disp_recv(size + 1, 0);
  std::partial_sum(number_recv.begin(), number_recv.end(),
                   disp_recv.begin() + 1);

  // Send/receive global indices
  std::vector<std::int64_t> vertices_recv(disp_recv.back());
  MPI_Alltoallv(indices_send.data(), number_send.data(), disp_send.data(),
                MPI::mpi_type<std::int64_t>(), vertices_recv.data(),
                number_recv.data(), disp_recv.data(),
                MPI::mpi_type<std::int64_t>(), comm);

  // Build list of sharing processes for each vertex
  const std::array range
      = dolfinx::MPI::local_range(rank, max_global_index + 1, size);
  std::vector<std::set<int>> owners(range[1] - range[0]);
  for (int i = 0; i < size; ++i)
  {
    assert((i + 1) < (int)disp_recv.size());
    for (int j = disp_recv[i]; j < disp_recv[i + 1]; ++j)
    {
      // Get back to 'zero' reference index
      assert(j < (int)vertices_recv.size());
      const std::int64_t index = vertices_recv[j] - range[0];

      assert(index < (int)owners.size());
      owners[index].insert(i);
    }
  }

  // For each index, build list of sharing processes
  std::unordered_map<std::int64_t, std::set<int>> global_vertex_to_procs;
  for (int i = 0; i < size; ++i)
  {
    for (int j = disp_recv[i]; j < disp_recv[i + 1]; ++j)
      global_vertex_to_procs[vertices_recv[j]].insert(i);
  }

  // For vertices on this process, get list of sharing process
  std::unique_ptr<const graph::AdjacencyList<int>> sharing_processes;
  {
    // Pack process that share each vertex
    std::vector<int> data_send, disp_send(size + 1, 0), num_send(size);
    for (int p = 0; p < size; ++p)
    {
      for (int j = disp_recv[p]; j < disp_recv[p + 1]; ++j)
      {
        const std::int64_t vertex = vertices_recv[j];
        auto it = global_vertex_to_procs.find(vertex);
        assert(it != global_vertex_to_procs.end());
        data_send.push_back(it->second.size());
        data_send.insert(data_send.end(), it->second.begin(), it->second.end());
      }
      disp_send[p + 1] = data_send.size();
      num_send[p] = disp_send[p + 1] - disp_send[p];
    }

    // Send/receive number of process that 'share' each vertex
    std::vector<int> num_recv(size);
    MPI_Alltoall(num_send.data(), 1, MPI_INT, num_recv.data(), 1, MPI_INT,
                 comm);

    // Compute receive displacements
    std::vector<int> disp_recv(size + 1, 0);
    std::partial_sum(num_recv.begin(), num_recv.end(), disp_recv.begin() + 1);

    // Send/receive sharing data
    std::vector<int> data_recv(disp_recv.back(), -1);
    MPI_Alltoallv(data_send.data(), num_send.data(), disp_send.data(),
                  MPI::mpi_type<int>(), data_recv.data(), num_recv.data(),
                  disp_recv.data(), MPI::mpi_type<int>(), comm);

    // Unpack data
    std::vector<int> processes, process_offsets(1, 0);
    for (std::size_t p = 0; p < disp_recv.size() - 1; ++p)
    {
      for (int i = disp_recv[p]; i < disp_recv[p + 1];)
      {
        const int num_procs = data_recv[i++];
        for (int j = 0; j < num_procs; ++j)
          processes.push_back(data_recv[i++]);
        process_offsets.push_back(process_offsets.back() + num_procs);
      }
    }

    sharing_processes = std::make_unique<const graph::AdjacencyList<int>>(
        processes, process_offsets);
  }

  // Build global-to-local map for non-shared indices (0)
  std::unordered_map<std::int64_t, std::int32_t> global_to_local_owned0;
  for (auto& vertex : global_to_local)
  {
    if (!shared_indices[vertex.second])
      global_to_local_owned0.insert(vertex);
  }

  // Loop over indices that were communicated and:
  // 1. Add 'exterior' but non-shared indices to global_to_local_owned0
  // 2. Add shared and owned indices to global_to_local_owned1
  // 3. Add non owned indices to global_to_local_unowned
  std::unordered_map<std::int64_t, std::int32_t> global_to_local_owned1,
      global_to_local_unowned;
  for (int i = 0; i < sharing_processes->num_nodes(); ++i)
  {
    auto it = global_to_local.find(indices_send[i]);
    assert(it != global_to_local.end());
    if (sharing_processes->num_links(i) == 1)
      global_to_local_owned0.insert(*it);
    else if (sharing_processes->links(i).minCoeff() == rank)
      global_to_local_owned1.insert(*it);
    else
      global_to_local_unowned.insert(*it);
  }

  // Re-number indices owned by this rank
  std::vector<std::int64_t> local_to_original;
  std::vector<std::int32_t> local_to_local_new(shared_indices.size(), -1);
  std::int32_t p = 0;
  for (const auto& index : global_to_local_owned0)
  {
    assert(index.second < (int)local_to_local_new.size());
    local_to_original.push_back(index.first);
    local_to_local_new[index.second] = p++;
  }
  for (const auto& index : global_to_local_owned1)
  {
    assert(index.second < (int)local_to_local_new.size());
    local_to_original.push_back(index.first);
    local_to_local_new[index.second] = p++;
  }

  // Compute process offset
  const std::int64_t num_owned_vertices
      = global_to_local_owned0.size() + global_to_local_owned1.size();
  const std::int64_t offset_global
      = dolfinx::MPI::global_offset(comm, num_owned_vertices, true);

  // Send global new global indices that this process has numbered
  for (int i = 0; i < sharing_processes->num_nodes(); ++i)
  {
    // Get old global -> local
    auto it = global_to_local.find(indices_send[i]);
    assert(it != global_to_local.end());
    if (sharing_processes->num_links(i) == 1
        and sharing_processes->links(i).minCoeff() == rank)
    {
      global_to_local_owned1.insert(*it);
    }
  }

  // Get array of unique neighboring process ranks, and remove self
  const Eigen::Array<int, Eigen::Dynamic, 1>& procs
      = sharing_processes->array();
  std::vector<int> neighbors(procs.data(), procs.data() + procs.rows());
  std::sort(neighbors.begin(), neighbors.end());
  neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                  neighbors.end());
  if (auto it = std::find(neighbors.begin(), neighbors.end(), rank);
      it != neighbors.end())
  {
    neighbors.erase(it);
  }

  // Create neighborhood communicator
  MPI_Comm comm_n;
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &comm_n);

  // Compute number on (global old, global new) pairs to send to each
  // neighbor
  std::vector<int> number_send_neigh(neighbors.size(), 0);
  for (std::size_t i = 0; i < neighbors.size(); ++i)
  {
    for (int j = 0; j < sharing_processes->num_nodes(); ++j)
    {
      auto p = sharing_processes->links(j);
      const auto* it = std::find(p.data(), p.data() + p.rows(), neighbors[i]);
      if (it != (p.data() + p.rows()))
        number_send_neigh[i] += 2;
    }
  }

  // Compute send displacements
  std::vector<int> disp_send_neigh(neighbors.size() + 1, 0);
  std::partial_sum(number_send_neigh.begin(), number_send_neigh.end(),
                   disp_send_neigh.begin() + 1);

  // Communicate number of values to send/receive
  std::vector<int> num_indices_recv(neighbors.size());
  MPI_Neighbor_alltoall(number_send_neigh.data(), 1, MPI_INT,
                        num_indices_recv.data(), 1, MPI_INT, comm_n);

  // Compute receive displacements
  std::vector<int> disp_recv_neigh(neighbors.size() + 1, 0);
  std::partial_sum(num_indices_recv.begin(), num_indices_recv.end(),
                   disp_recv_neigh.begin() + 1);

  // Pack data to send
  std::vector<int> offset_neigh = disp_send_neigh;
  std::vector<std::int64_t> data_send_neigh(disp_send_neigh.back(), -1);
  for (std::size_t p = 0; p < neighbors.size(); ++p)
  {
    const int neighbor = neighbors[p];
    for (int i = 0; i < sharing_processes->num_nodes(); ++i)
    {
      auto it = global_to_local.find(indices_send[i]);
      assert(it != global_to_local.end());

      const std::int64_t global_old = it->first;
      const std::int32_t local_old = it->second;
      std::int64_t global_new = local_to_local_new[local_old];
      if (global_new >= 0)
        global_new += offset_global;

      auto procs = sharing_processes->links(i);
      for (int k = 0; k < procs.rows(); ++k)
      {
        if (procs[k] == neighbor)
        {
          data_send_neigh[offset_neigh[p]++] = global_old;
          data_send_neigh[offset_neigh[p]++] = global_new;
        }
      }
    }
  }

  // Send/receive data
  std::vector<std::int64_t> data_recv_neigh(disp_recv_neigh.back());
  MPI_Neighbor_alltoallv(data_send_neigh.data(), number_send_neigh.data(),
                         disp_send_neigh.data(), MPI_INT64_T,
                         data_recv_neigh.data(), num_indices_recv.data(),
                         disp_recv_neigh.data(), MPI_INT64_T, comm_n);

  MPI_Comm_free(&comm_n);

  // Unpack received (global old, global new) pairs
  std::map<std::int64_t, std::pair<std::int64_t, int>> global_old_new;
  for (std::size_t i = 0; i < data_recv_neigh.size(); i += 2)
  {
    if (data_recv_neigh[i + 1] >= 0)
    {
      const auto pos = std::upper_bound(disp_recv_neigh.begin(),
                                        disp_recv_neigh.end(), i + 1);
      const int owner = std::distance(disp_recv_neigh.begin(), pos) - 1;
      global_old_new.insert(
          {data_recv_neigh[i], {data_recv_neigh[i + 1], neighbors[owner]}});
    }
  }

  // Build array of ghost indices (indices owned and numbered by another
  // process)
  std::vector<std::int64_t> ghosts;
  std::vector<int> ghost_owners;
  for (auto it = global_to_local_unowned.begin();
       it != global_to_local_unowned.end(); ++it)
  {
    if (auto pair = global_old_new.find(it->first);
        pair != global_old_new.end())
    {
      assert(it->second < (int)local_to_local_new.size());
      local_to_original.push_back(it->first);
      local_to_local_new[it->second] = p++;
      ghosts.push_back(pair->second.first);
      ghost_owners.push_back(pair->second.second);
    }
  }

  return {std::move(local_to_local_new), std::move(ghosts),
          std::move(ghost_owners)};
}
//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>>
Partitioning::create_local_adjacency_list(
    const graph::AdjacencyList<std::int64_t>& cells)
{
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& array = cells.array();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> array_local(array.rows());

  // Re-map global to local
  int local = 0;
  std::unordered_map<std::int64_t, std::int32_t> global_to_local;
  for (int i = 0; i < array.rows(); ++i)
  {
    const std::int64_t global = array(i);
    const auto [it, inserted] = global_to_local.insert({global, local});
    if (inserted)
    {
      array_local[i] = local;
      ++local;
    }
    else
      array_local[i] = it->second;
  }

  std::vector<std::int64_t> local_to_global(global_to_local.size());
  for (const auto& e : global_to_local)
    local_to_global[e.second] = e.first;

  return {graph::AdjacencyList<std::int32_t>(std::move(array_local),
                                             cells.offsets()),
          std::move(local_to_global)};
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int32_t>, common::IndexMap>
Partitioning::create_distributed_adjacency_list(
    MPI_Comm comm, const graph::AdjacencyList<std::int32_t>& list_local,
    const std::vector<std::int64_t>& local_to_global_links,
    const std::vector<bool>& shared_links)
{
  common::Timer timer("Create distributed AdjacencyList");

  // Compute new local and global indices
  const auto [local_to_local_new, ghosts, ghost_owners]
      = reorder_global_indices(comm, local_to_global_links, shared_links);

  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& data_old
      = list_local.array();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> data_new(data_old.rows());
  for (int i = 0; i < data_new.rows(); ++i)
    data_new[i] = local_to_local_new[data_old[i]];

  const int num_owned_vertices = local_to_local_new.size() - ghosts.size();
  return {graph::AdjacencyList<std::int32_t>(std::move(data_new),
                                             list_local.offsets()),
          common::IndexMap(comm, num_owned_vertices,
                           dolfinx::MPI::compute_graph_edges(
                               comm, std::set<int>(ghost_owners.begin(),
                                                   ghost_owners.end())),
                           ghosts, ghost_owners, 1)};
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
Partitioning::distribute(MPI_Comm comm,
                         const graph::AdjacencyList<std::int64_t>& list,
                         const graph::AdjacencyList<std::int32_t>& destinations)
{
  common::Timer timer("Distribute AdjacencyList");

  assert(list.num_nodes() == (int)destinations.num_nodes());
  const std::int64_t offset_global
      = dolfinx::MPI::global_offset(comm, list.num_nodes(), true);

  const int size = dolfinx::MPI::size(comm);
  // Compute number of links to send to each process
  std::vector<int> num_per_dest_send(size, 0);
  for (int i = 0; i < destinations.num_nodes(); ++i)
  {
    const auto& dests = destinations.links(i);
    for (int j = 0; j < destinations.num_links(i); ++j)
      num_per_dest_send[dests[j]] += list.num_links(i) + 3;
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

  // Prepare send buffer
  std::vector<int> offset = disp_send;
  std::vector<std::int64_t> data_send(disp_send.back());
  for (int i = 0; i < list.num_nodes(); ++i)
  {
    const auto& dests = destinations.links(i);
    for (std::int32_t j = 0; j < destinations.num_links(i); ++j)
    {
      std::int32_t dest = dests[j];
      auto links = list.links(i);
      data_send[offset[dest]++] = dests[0];
      data_send[offset[dest]++] = i + offset_global;
      data_send[offset[dest]++] = links.rows();
      for (int k = 0; k < links.rows(); ++k)
        data_send[offset[dest]++] = links(k);
    }
  }

  // Send/receive data
  std::vector<std::int64_t> data_recv(disp_recv.back());
  MPI_Alltoallv(data_send.data(), num_per_dest_send.data(), disp_send.data(),
                MPI_INT64_T, data_recv.data(), num_per_dest_recv.data(),
                disp_recv.data(), MPI_INT64_T, comm);

  // Unpack receive buffer
  int mpi_rank = MPI::rank(comm);
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
        for (int j = 0; j < num_links; ++j)
          array.push_back(data_recv[i++]);
        list_offset.push_back(list_offset.back() + num_links);
      }
      else
      {
        ghost_src.push_back(p);
        ghost_index_owner.push_back(data_recv[i++]);
        ghost_global_indices.push_back(data_recv[i++]);
        const std::int64_t num_links = data_recv[i++];
        for (int j = 0; j < num_links; ++j)
          ghost_array.push_back(data_recv[i++]);
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
  for (int& offset : ghost_list_offset)
    offset += ghost_offset;
  list_offset.insert(list_offset.end(), ghost_list_offset.begin(),
                     ghost_list_offset.end());

  return {graph::AdjacencyList<std::int64_t>(array, list_offset),
          std::move(src), std::move(global_indices),
          std::move(ghost_index_owner)};
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> Partitioning::compute_ghost_indices(
    MPI_Comm comm, const std::vector<std::int64_t>& global_indices,
    const std::vector<int>& ghost_owners)
{
  LOG(INFO) << "Compute ghost indices";

  // Get number of local cells and global offset
  int num_local = global_indices.size() - ghost_owners.size();
  std::vector<std::int64_t> ghost_global_indices(
      global_indices.begin() + num_local, global_indices.end());

  const std::int64_t offset_local
      = dolfinx::MPI::global_offset(comm, num_local, true);

  // Find out how many ghosts are on each neighboring process
  std::vector<int> ghost_index_count;
  std::vector<int> neighbors;
  std::map<int, int> proc_to_neighbor;
  int np = 0;
  int mpi_rank = MPI::rank(comm);
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
  for (std::size_t i = 0; i < ghost_index_count.size(); ++i)
    send_offsets.push_back(send_offsets.back() + ghost_index_count[i]);
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
  MPI_Neighbor_alltoall(ghost_index_count.data(), 1, MPI_INT, recv_sizes.data(),
                        1, MPI_INT, neighbor_comm);
  std::vector<int> recv_offsets = {0};
  for (int q : recv_sizes)
    recv_offsets.push_back(recv_offsets.back() + q);
  std::vector<std::int64_t> recv_data(recv_offsets.back());

  MPI_Neighbor_alltoallv(send_data.data(), ghost_index_count.data(),
                         send_offsets.data(), MPI_INT64_T, recv_data.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT64_T,
                         neighbor_comm);

  // Replace values in recv_data with new_index and send back
  std::unordered_map<std::int64_t, std::int64_t> old_to_new;
  for (int i = 0; i < num_local; ++i)
    old_to_new.insert({global_indices[i], offset_local + i});

  for (std::int64_t& r : recv_data)
  {
    auto it = old_to_new.find(r);
    // Must exist on this process!
    assert(it != old_to_new.end());
    r = it->second;
  }

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
    auto [it, insert] = old_to_new.insert({old_idx, new_idx});
    assert(insert);
  }

  for (std::int64_t& q : ghost_global_indices)
  {
    const auto it = old_to_new.find(q);
    assert(it != old_to_new.end());
    q = it->second;
  }

  MPI_Comm_free(&neighbor_comm);
  return ghost_global_indices;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> Partitioning::compute_local_to_global_links(
    const graph::AdjacencyList<std::int64_t>& global,
    const graph::AdjacencyList<std::int32_t>& local)
{
  common::Timer timer(
      "Compute-local-to-global links for global/local adjacency list");

  // Build local-to-global for adjacency lists
  if (global.num_nodes() != local.num_nodes())
  {
    throw std::runtime_error("Mismatch in number of nodes between local and "
                             "global adjacency lists.");
  }

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& _global = global.array();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& _local = local.array();
  if (_global.rows() != _local.rows())
  {
    throw std::runtime_error("Data size mismatch between local and "
                             "global adjacency lists.");
  }

  const std::int32_t max_local = _local.maxCoeff();
  std::vector<bool> marker(max_local, false);
  std::vector<std::int64_t> local_to_global_list(max_local + 1, -1);
  for (Eigen::Index i = 0; i < _local.rows(); ++i)
  {
    if (local_to_global_list[_local[i]] == -1)
      local_to_global_list[_local[i]] = _global[i];
  }

  return local_to_global_list;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> Partitioning::compute_local_to_local(
    const std::vector<std::int64_t>& local0_to_global,
    const std::vector<std::int64_t>& local1_to_global)
{
  common::Timer timer("Compute local-to-local map");
  assert(local0_to_global.size() == local1_to_global.size());

  // Compute inverse map for local1_to_global
  std::unordered_map<std::int64_t, std::int32_t> global_to_local1;
  for (std::size_t i = 0; i < local1_to_global.size(); ++i)
    global_to_local1.insert({local1_to_global[i], i});

  // Compute inverse map for local0_to_local1
  std::vector<std::int32_t> local0_to_local1(local0_to_global.size());
  for (std::size_t i = 0; i < local0_to_local1.size(); ++i)
  {
    auto it = global_to_local1.find(local0_to_global[i]);
    assert(it != global_to_local1.end());
    local0_to_local1[i] = it->second;
  }

  return local0_to_local1;
}
//-----------------------------------------------------------------------------
