// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PartitioningNew.h"
#include "Topology.h"
#include <Eigen/Dense>
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/CSRGraph.h>
#include <dolfinx/graph/GraphBuilder.h>
#include <dolfinx/graph/SCOTCH.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
// std::array<std::vector<std::int32_t>, 2>
std::vector<bool> PartitioningNew::compute_vertex_exterior_markers(
    const mesh::Topology& topology_local)
{
  // Get list of boundary vertices
  const int dim = topology_local.dim();
  auto facet_cell = topology_local.connectivity(dim - 1, dim);
  if (!facet_cell)
  {
    throw std::runtime_error(
        "Need facet-cell connectivity to build distributed adjacency list.");
  }

  auto facet_vertex = topology_local.connectivity(dim - 1, 0);
  if (!facet_vertex)
  {
    throw std::runtime_error(
        "Need facet-vertex connectivity to build distributed adjacency list.");
  }

  auto map_vertex = topology_local.index_map(0);
  if (!map_vertex)
    throw std::runtime_error("Need vertex IndexMap from topology.");
  assert(map_vertex->num_ghosts() == 0);

  std::vector<bool> exterior_vertex(map_vertex->size_local(), false);
  for (int f = 0; f < facet_cell->num_nodes(); ++f)
  {
    if (facet_cell->num_links(f) == 1)
    {
      auto vertices = facet_vertex->links(f);
      for (int j = 0; j < vertices.rows(); ++j)
        exterior_vertex[vertices[j]] = true;
    }
  }

  return exterior_vertex;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<std::int32_t>, std::vector<std::int64_t>>
PartitioningNew::reorder_global_indices(
    MPI_Comm comm, const std::vector<std::int64_t>& global_indices,
    const std::vector<bool>& shared_indices)
{
  // TODO: Can this function be broken into multiple logical steps?

  assert(global_indices.size() == shared_indices.size());

  // Create global ->local map
  std::map<std::int64_t, std::int32_t> global_to_local;
  for (std::size_t i = 0; i < global_indices.size(); ++i)
    global_to_local.insert({global_indices[i], i});

  // Get maximum global index across all processes
  std::int64_t my_max_global_index = 0;
  if (!global_to_local.empty())
    my_max_global_index = global_to_local.rbegin()->first;
  const std::int64_t max_global_index
      = dolfinx::MPI::all_reduce(comm, my_max_global_index, MPI_MAX);

  // Compute number of possibly shared vertices to send to each process,
  // considering only vertices that are possibly shared
  const int size = dolfinx::MPI::size(comm);
  std::vector<int> number_send(size, 0);
  for (auto& vertex : global_to_local)
  {
    if (shared_indices[vertex.second])
    {
      // TODO: optimise this call
      const int owner
          = dolfinx::MPI::index_owner(comm, vertex.first, max_global_index + 1);
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
          = dolfinx::MPI::index_owner(comm, vertex.first, max_global_index + 1);
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
  const std::array<std::int64_t, 2> range
      = dolfinx::MPI::local_range(comm, max_global_index + 1);
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
  std::map<std::int64_t, std::set<int>> global_vertex_to_procs;
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
  std::map<std::int64_t, std::int32_t> global_to_local_owned0;
  for (auto& vertex : global_to_local)
  {
    if (!shared_indices[vertex.second])
      global_to_local_owned0.insert(vertex);
  }

  // Loop over indices that were communicated and:
  // 1. Add 'exterior' but non-shared indices to global_to_local_owned0
  // 2. Add shared and owned indices to global_to_local_owned1
  // 3. Add non owned indices to global_to_local_unowned
  const int rank = dolfinx::MPI::rank(comm);
  std::map<std::int64_t, std::int32_t> global_to_local_owned1,
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
    // if (sharing_processes->num_links(i) == 1)
    //   global_to_local_owned0.insert(*it);
    // else if (sharing_processes->links(i).minCoeff() == rank)
    //   global_to_local_owned1.insert(*it);
  }

  // Get array of unique neighbouring process ranks, and remove self
  const Eigen::Array<int, Eigen::Dynamic, 1>& procs
      = sharing_processes->array();
  std::vector<int> neighbours(procs.data(), procs.data() + procs.rows());
  std::sort(neighbours.begin(), neighbours.end());
  neighbours.erase(std::unique(neighbours.begin(), neighbours.end()),
                   neighbours.end());
  auto it = std::find(neighbours.begin(), neighbours.end(), rank);
  neighbours.erase(it);

  // Create neighbourhood communicator
  MPI_Comm comm_n;
  MPI_Dist_graph_create_adjacent(comm, neighbours.size(), neighbours.data(),
                                 MPI_UNWEIGHTED, neighbours.size(),
                                 neighbours.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &comm_n);

  // Compute number on (global old, global new) pairs to send to each
  // neighbour
  std::vector<int> number_send_neigh(neighbours.size(), 0);
  for (std::size_t i = 0; i < neighbours.size(); ++i)
  {
    for (int j = 0; j < sharing_processes->num_nodes(); ++j)
    {
      auto p = sharing_processes->links(j);
      auto it = std::find(p.data(), p.data() + p.rows(), neighbours[i]);
      if (it != (p.data() + p.rows()))
        number_send_neigh[i] += 2;
    }
  }

  // Compute send displacements
  std::vector<int> disp_send_neigh(neighbours.size() + 1, 0);
  std::partial_sum(number_send_neigh.begin(), number_send_neigh.end(),
                   disp_send_neigh.begin() + 1);

  // Communicate number of values to send/receive
  std::vector<int> num_indices_recv(neighbours.size());
  MPI_Neighbor_alltoall(number_send_neigh.data(), 1, MPI_INT,
                        num_indices_recv.data(), 1, MPI_INT, comm_n);

  // Compute receive displacements
  std::vector<int> disp_recv_neigh(neighbours.size() + 1, 0);
  std::partial_sum(num_indices_recv.begin(), num_indices_recv.end(),
                   disp_recv_neigh.begin() + 1);

  // Pack data to send
  std::vector<int> offset_neigh = disp_send_neigh;
  std::vector<std::int64_t> data_send_neigh(disp_send_neigh.back(), -1);
  for (std::size_t p = 0; p < neighbours.size(); ++p)
  {
    const int neighbour = neighbours[p];
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
        if (procs[k] == neighbour)
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
  std::map<std::int64_t, std::int64_t> global_old_new;
  for (std::size_t i = 0; i < data_recv_neigh.size(); i += 2)
  {
    if (data_recv_neigh[i + 1] >= 0)
      global_old_new.insert({data_recv_neigh[i], data_recv_neigh[i + 1]});
  }

  // Build array of ghost indices (indices owned and numbered by another
  // process)
  std::vector<std::int64_t> ghosts;
  for (auto it = global_to_local_unowned.begin();
       it != global_to_local_unowned.end(); ++it)
  {
    auto pair = global_old_new.find(it->first);
    if (pair != global_old_new.end())
    {
      assert(it->second < (int)local_to_local_new.size());
      local_to_original.push_back(it->first);
      local_to_local_new[it->second] = p++;
      ghosts.push_back(pair->second);
    }
  }

  return {local_to_local_new, ghosts};
}
//-----------------------------------------------------------------------------
std::vector<int> PartitioningNew::partition_cells(
    MPI_Comm comm, int nparts, const mesh::CellType cell_type,
    const graph::AdjacencyList<std::int64_t>& cells)
{
  LOG(INFO) << "Compute partition of cells across processes";

  // FIXME: Update GraphBuilder to use AdjacencyList
  // Wrap AdjacencyList
  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
      _cells(cells.array().data(), cells.num_nodes(),
             mesh::num_cell_vertices(cell_type));

  // Compute dual graph (for the cells on this process)
  const auto [local_graph, graph_info]
      = graph::GraphBuilder::compute_dual_graph(comm, _cells, cell_type);
  const auto [num_ghost_nodes, num_local_edges, num_nonlocal_edges]
      = graph_info;

  // Build graph
  graph::CSRGraph<SCOTCH_Num> csr_graph(comm, local_graph);
  std::vector<std::size_t> weights;

  // Call partitioner
  const auto [partition, ignore] = graph::SCOTCH::partition(
      comm, (SCOTCH_Num)nparts, csr_graph, weights, num_ghost_nodes);

  return partition;
}
//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>>
PartitioningNew::create_local_adjacency_list(
    const graph::AdjacencyList<std::int64_t>& cells)
{
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& array = cells.array();
  std::vector<std::int32_t> array_local(array.rows());

  // Re-map global to local
  int local = 0;
  std::map<std::int64_t, std::int32_t> global_to_local;
  for (int i = 0; i < array.rows(); ++i)
  {
    const std::int64_t global = array(i);
    auto it = global_to_local.find(global);
    if (it == global_to_local.end())
    {
      array_local[i] = local;
      global_to_local.insert({global, local});
      ++local;
    }
    else
      array_local[i] = it->second;
  }

  std::vector<std::int64_t> local_to_global(global_to_local.size());
  for (const auto& e : global_to_local)
    local_to_global[e.second] = e.first;

  // FIXME: Update AdjacencyList to avoid this
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& offsets
      = cells.offsets();
  std::vector<std::int32_t> _offsets(offsets.data(),
                                     offsets.data() + offsets.rows());
  return {graph::AdjacencyList<std::int32_t>(array_local, _offsets),
          std::move(local_to_global)};
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int32_t>, common::IndexMap>
PartitioningNew::create_distributed_adjacency_list(
    MPI_Comm comm, const mesh::Topology& topology_local,
    const std::vector<std::int64_t>& local_to_global_vertices)
{
  // Get marker for each vertex indicating if it interior or on the
  // boundary of the local topology
  const std::vector<bool>& exterior_vertex
      = compute_vertex_exterior_markers(topology_local);

  // Compute new local and global indices
  const auto [local_to_local_new, ghosts]
      = reorder_global_indices(comm, local_to_global_vertices, exterior_vertex);

  const int dim = topology_local.dim();
  auto cv = topology_local.connectivity(dim, 0);
  if (!cv)
    throw std::runtime_error("Missing cell-vertex connectivity.");

  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& data_old = cv->array();
  std::vector<std::int32_t> data_new(data_old.rows());
  for (std::size_t i = 0; i < data_new.size(); ++i)
    data_new[i] = local_to_local_new[data_old[i]];

  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& offsets = cv->offsets();
  std::vector<std::int32_t> _offsets(offsets.data(),
                                     offsets.data() + offsets.rows());

  const int num_owned_vertices = local_to_local_new.size() - ghosts.size();
  return {graph::AdjacencyList<std::int32_t>(data_new, _offsets),
          common::IndexMap(comm, num_owned_vertices, ghosts, 1)};
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>>
PartitioningNew::distribute(MPI_Comm comm,
                            const graph::AdjacencyList<std::int64_t>& list,
                            const std::vector<int>& destinations)
{
  assert(list.num_nodes() == (int)destinations.size());
  const std::int64_t offset_global
      = dolfinx::MPI::global_offset(comm, destinations.size(), true);

  const int size = dolfinx::MPI::size(comm);

  // Compute number of links to send to each process
  std::vector<int> num_per_dest_send(size, 0);
  assert(list.num_nodes() == (int)destinations.size());
  for (int i = 0; i < list.num_nodes(); ++i)
    num_per_dest_send[destinations[i]] += list.num_links(i) + 2;

  // Compute send array displacements
  std::vector<int> disp_send(size + 1, 0);
  std::partial_sum(num_per_dest_send.begin(), num_per_dest_send.end(),
                   disp_send.begin() + 1);

  // Send/receive number of items to communicate
  std::vector<int> num_per_dest_recv(size, 0);
  MPI_Alltoall(num_per_dest_send.data(), 1, MPI_INT, num_per_dest_recv.data(),
               1, MPI_INT, comm);

  // Compite receive array displacements
  std::vector<int> disp_recv(size + 1, 0);
  std::partial_sum(num_per_dest_recv.begin(), num_per_dest_recv.end(),
                   disp_recv.begin() + 1);

  // Prepare send buffer
  std::vector<int> offset = disp_send;
  std::vector<std::int64_t> data_send(disp_send.back());
  for (int i = 0; i < list.num_nodes(); ++i)
  {
    const int dest = destinations[i];
    auto links = list.links(i);
    data_send[offset[dest]++] = i + offset_global;
    data_send[offset[dest]++] = links.rows();
    for (int j = 0; j < links.rows(); ++j)
      data_send[offset[dest]++] = links(j);
  }

  // Send/receive data
  std::vector<std::int64_t> data_recv(disp_recv.back());
  MPI_Alltoallv(data_send.data(), num_per_dest_send.data(), disp_send.data(),
                MPI_INT64_T, data_recv.data(), num_per_dest_recv.data(),
                disp_recv.data(), MPI_INT64_T, comm);

  // Unpack receive buffer
  std::vector<std::int64_t> array, global_indices;
  std::vector<std::int32_t> list_offset(1, 0);
  std::vector<int> src;
  for (std::size_t p = 0; p < disp_recv.size() - 1; ++p)
  {
    for (int i = disp_recv[p]; i < disp_recv[p + 1];)
    {
      src.push_back(p);
      global_indices.push_back(data_recv[i++]);
      const std::int64_t num_links = data_recv[i++];
      for (int j = 0; j < num_links; ++j)
        array.push_back(data_recv[i++]);
      list_offset.push_back(list_offset.back() + num_links);
    }
  }

  return {graph::AdjacencyList<std::int64_t>(array, list_offset),
          std::move(src), std::move(global_indices)};
}
//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int64_t>, std::vector<std::int64_t>>
PartitioningNew::exchange(MPI_Comm comm,
                          const graph::AdjacencyList<std::int64_t>& list,
                          const std::vector<int>& destinations,
                          const std::set<int>&)
{
  // TODO: This can be significantly optimised (avoiding all-to-all) by
  // sending in more information on source/dest ranks

  assert(list.num_nodes() == (int)destinations.size());

  const std::int64_t offset_global
      = dolfinx::MPI::global_offset(comm, destinations.size(), true);

  const int size = dolfinx::MPI::size(comm);

  // Compute number of links to send to each process
  std::vector<int> num_per_dest_send(size, 0);
  assert(list.num_nodes() == (int)destinations.size());
  for (int i = 0; i < list.num_nodes(); ++i)
    num_per_dest_send[destinations[i]] += list.num_links(i) + 2;

  // Compute send array displacements
  std::vector<int> disp_send(size + 1, 0);
  std::partial_sum(num_per_dest_send.begin(), num_per_dest_send.end(),
                   disp_send.begin() + 1);

  // Send/receive number of items to communicate
  std::vector<int> num_per_dest_recv(size, 0);
  MPI_Alltoall(num_per_dest_send.data(), 1, MPI_INT, num_per_dest_recv.data(),
               1, MPI_INT, comm);

  // Compite receive array displacements
  std::vector<int> disp_recv(size + 1, 0);
  std::partial_sum(num_per_dest_recv.begin(), num_per_dest_recv.end(),
                   disp_recv.begin() + 1);

  // Prepare send buffer
  std::vector<int> offset = disp_send;
  std::vector<std::int64_t> data_send(disp_send.back());
  for (int i = 0; i < list.num_nodes(); ++i)
  {
    const int dest = destinations[i];
    auto links = list.links(i);
    data_send[offset[dest]++] = i + offset_global;
    data_send[offset[dest]++] = links.rows();
    for (int j = 0; j < links.rows(); ++j)
      data_send[offset[dest]++] = links(j);
  }

  // Send/receive data
  std::vector<std::int64_t> data_recv(disp_recv.back());
  MPI_Alltoallv(data_send.data(), num_per_dest_send.data(), disp_send.data(),
                MPI_INT64_T, data_recv.data(), num_per_dest_recv.data(),
                disp_recv.data(), MPI_INT64_T, comm);

  // Unpack receive buffer
  std::vector<std::int64_t> array, global_indices;
  std::vector<std::int32_t> list_offset(1, 0);
  for (std::size_t p = 0; p < disp_recv.size() - 1; ++p)
  {
    for (int i = disp_recv[p]; i < disp_recv[p + 1];)
    {
      global_indices.push_back(data_recv[i++]);
      const std::int64_t num_links = data_recv[i++];
      for (int j = 0; j < num_links; ++j)
        array.push_back(data_recv[i++]);
      list_offset.push_back(list_offset.back() + num_links);
    }
  }

  return {graph::AdjacencyList<std::int64_t>(array, list_offset),
          std::move(global_indices)};
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
PartitioningNew::fetch_data(
    MPI_Comm comm, const std::vector<std::int64_t>& indices,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x)
{
  // Get number of points globally
  const std::int64_t num_points = dolfinx::MPI::sum(comm, x.rows());

  // Get ownership range for this rank, and compute offset
  const std::array<std::int64_t, 2> range
      = dolfinx::MPI::local_range(comm, num_points);
  const std::int64_t offset_x
      = dolfinx::MPI::global_offset(comm, range[1] - range[0], true);

  const int gdim = x.cols();
  assert(gdim != 0);
  const int size = dolfinx::MPI::size(comm);

  // Determine number of points to send to owner
  std::vector<int> number_send(size, 0);
  for (int i = 0; i < x.rows(); ++i)
  {
    // TODO: optimise this call
    const std::int64_t index_global = i + offset_x;
    const int owner = dolfinx::MPI::index_owner(comm, index_global, num_points);
    number_send[owner] += 1;
  }

  // Compute send displacements
  std::vector<int> disp_send(size + 1, 0);
  std::partial_sum(number_send.begin(), number_send.end(),
                   disp_send.begin() + 1);

  // Pack x data
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_send(
      disp_send.back(), gdim);
  std::vector<int> disp_tmp = disp_send;
  for (int i = 0; i < x.rows(); ++i)
  {
    const std::int64_t index_global = i + offset_x;
    const int owner = dolfinx::MPI::index_owner(comm, index_global, num_points);
    x_send.row(disp_tmp[owner]++) = x.row(i);
  }

  // Send/receive number of points to communicate to each process
  std::vector<int> number_recv(size);
  MPI_Alltoall(number_send.data(), 1, MPI_INT, number_recv.data(), 1, MPI_INT,
               comm);

  // Compute receive displacements
  std::vector<int> disp_recv(size + 1, 0);
  std::partial_sum(number_recv.begin(), number_recv.end(),
                   disp_recv.begin() + 1);

  // Build compound data type. This will allow us to re-use send/receive
  // displacements for indices and point data
  MPI_Datatype compound_f64;
  MPI_Type_contiguous(gdim, MPI_DOUBLE, &compound_f64);
  MPI_Type_commit(&compound_f64);

  // Send/receive points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_recv(
      disp_recv.back(), gdim);
  MPI_Alltoallv(x_send.data(), number_send.data(), disp_send.data(),
                compound_f64, x_recv.data(), number_recv.data(),
                disp_recv.data(), compound_f64, comm);

  // Build index data requests
  std::vector<int> number_index_send(size, 0);
  for (std::int64_t index : indices)
  {
    // TODO: optimise this call
    const int owner = dolfinx::MPI::index_owner(comm, index, num_points);
    number_index_send[owner] += 1;
  }

  // Compute send displacements
  std::vector<int> disp_index_send(size + 1, 0);
  std::partial_sum(number_index_send.begin(), number_index_send.end(),
                   disp_index_send.begin() + 1);

  // Pack global index send data
  std::vector<std::int64_t> indices_send(disp_index_send.back());
  disp_tmp = disp_index_send;
  for (std::int64_t index : indices)
  {
    // TODO: optimise this call
    const int owner = dolfinx::MPI::index_owner(comm, index, num_points);
    indices_send[disp_tmp[owner]++] = index;
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
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      x_return(indices_recv.size(), gdim);
  for (int p = 0; p < size; ++p)
  {
    for (int i = disp_index_recv[p]; i < disp_index_recv[p + 1]; ++i)
    {
      const std::int64_t index = indices_recv[i];
      const std::int32_t index_local = index - offset_x;
      assert(index_local >= 0);
      x_return.row(i) = x_recv.row(index_local);
    }
  }

  // Send back point data
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> my_x(
      disp_index_send.back(), gdim);
  MPI_Alltoallv(x_return.data(), number_index_recv.data(),
                disp_index_recv.data(), compound_f64, my_x.data(),
                number_index_send.data(), disp_index_send.data(), compound_f64,
                comm);

  return my_x;
}
//-----------------------------------------------------------------------------
