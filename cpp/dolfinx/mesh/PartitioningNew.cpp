// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PartitioningNew.h"
#include "Topology.h"
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
  // std::vector<std::int32_t> interior, exterior;
  // for (int i = 0; i < map_vertex->size_local(); ++i)
  // {
  //   if (exterior_vertex[i])
  //     exterior.push_back(i);
  //   else
  //     interior.push_back(i);
  // }

  // return {std::move(interior), std::move(exterior)};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<std::int32_t>, std::vector<std::int64_t>>
PartitioningNew::reorder_global_indices(
    MPI_Comm comm,
    const std::map<std::int64_t, std::int32_t>& global_to_local_vertices,
    const std::vector<bool>& shared_indices)
{
  assert(global_to_local_vertices.size() == shared_indices.size());

  // Get maximum global index across all processes
  std::int64_t my_max_global_index = 0;
  if (!global_to_local_vertices.empty())
    my_max_global_index = global_to_local_vertices.rbegin()->first;
  const std::int64_t max_global_index
      = dolfinx::MPI::all_reduce(comm, my_max_global_index, MPI_MAX);

  // Compute number of possibly shared vertices to send to each process,
  // considering only vertices that are possibly shared
  const int size = dolfinx::MPI::size(comm);
  std::vector<int> number_send(size, 0);
  for (auto& vertex : global_to_local_vertices)
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
  for (auto vertex : global_to_local_vertices)
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
    for (int i = 0; i < size; ++i)
    {
      for (int j = disp_recv[i]; j < disp_recv[i + 1]; ++j)
      {
        const std::int64_t vertex = vertices_recv[j];
        auto it = global_vertex_to_procs.find(vertex);
        assert(it != global_vertex_to_procs.end());
        data_send.push_back(it->second.size());
        data_send.insert(data_send.end(), it->second.begin(), it->second.end());
      }
      disp_send[i + 1] = data_send.size();
      num_send[i] = disp_send[i + 1] - disp_send[i];
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
      for (int i = disp_recv[p]; i < disp_recv[p + 1]; ++i)
      {
        const int num_procs = data_recv[i];
        const int pos = i;
        for (int j = 1; j <= num_procs; ++j)
        {
          processes.push_back(data_recv[pos + j]);
          i += 1;
        }
        process_offsets.push_back(process_offsets.back() + num_procs);
      }
    }

    sharing_processes = std::make_unique<const graph::AdjacencyList<int>>(
        processes, process_offsets);
  }

  // Build global-to-local map for non-shared indices
  std::map<std::int64_t, std::int32_t> global_to_local_owned0;
  for (auto& vertex : global_to_local_vertices)
  {
    if (!shared_indices[vertex.second])
      global_to_local_owned0.insert(vertex);
  }

  // Build global-to-local map for (i) 'exterior' but non-shared indices
  // and (ii) shared indices
  const int rank = dolfinx::MPI::rank(comm);
  std::map<std::int64_t, std::int32_t> global_to_local_owned1,
      global_to_local_unowned;
  for (int i = 0; i < sharing_processes->num_nodes(); ++i)
  {
    auto it = global_to_local_vertices.find(indices_send[i]);
    assert(it != global_to_local_vertices.end());
    if (sharing_processes->num_links(i) == 1)
      global_to_local_owned0.insert(*it);
    else if (sharing_processes->links(i).minCoeff() == rank)
      global_to_local_owned1.insert(*it);
    else
      global_to_local_unowned.insert(*it);
  }

  // Re-number owned indices
  std::vector<std::int64_t> local_to_original;
  std::vector<std::int32_t> local_to_local_new(shared_indices.size(), -1);
  std::int32_t p = 0;
  for (auto it = global_to_local_owned0.begin();
       it != global_to_local_owned0.end(); ++it)
  {
    assert(it->second < (int)local_to_local_new.size());
    local_to_original.push_back(it->first);
    local_to_local_new[it->second] = p++;
  }
  for (auto it = global_to_local_owned1.begin();
       it != global_to_local_owned1.end(); ++it)
  {
    assert(it->second < (int)local_to_local_new.size());
    local_to_original.push_back(it->first);
    local_to_local_new[it->second] = p++;
  }

  // Compute process offset
  const std::int64_t num_owned_vertices
      = global_to_local_owned0.size() + global_to_local_owned1.size();
  const std::int64_t offset_global
      = dolfinx::MPI::global_offset(comm, num_owned_vertices, true);

  // Send global new global indices
  for (int i = 0; i < sharing_processes->num_nodes(); ++i)
  {
    // Get old global -> local
    auto it = global_to_local_vertices.find(indices_send[i]);
    assert(it != global_to_local_vertices.end());

    if (sharing_processes->num_links(i) == 1)
      global_to_local_owned0.insert(*it);
    else if (sharing_processes->links(i).minCoeff() == rank)
      global_to_local_owned1.insert(*it);
  }

  // Get array of unique neighbouring process ranks
  const Eigen::Array<int, Eigen::Dynamic, 1>& procs
      = sharing_processes->array();
  std::vector<int> neighbours(procs.data(), procs.data() + procs.rows());
  std::sort(neighbours.begin(), neighbours.end());
  neighbours.erase(std::unique(neighbours.begin(), neighbours.end()),
                   neighbours.end());

  // Number of (global old, global new) pairs to send to each neighbour
  std::map<int, int> num_receive;
  for (int p : neighbours)
  {
    num_receive[p]
        = 2 * std::count(procs.data(), procs.data() + procs.rows(), p);
  }

  // Send (global old, global new) pairs to neighbours
  const int num_neighbours = neighbours.size();
  std::vector<MPI_Request> requests(2 * num_neighbours);
  std::vector<std::vector<int>> dsend(neighbours.size());
  for (int p = 0; p < num_neighbours; ++p)
  {
    // std::vector<int> dsend;
    for (int i = 0; i < sharing_processes->num_nodes(); ++i)
    {
      auto it = global_to_local_vertices.find(indices_send[i]);
      assert(it != global_to_local_vertices.end());

      const std::int64_t global_old = it->first;
      const std::int32_t local_old = it->second;
      std::int64_t global_new = local_to_local_new[local_old];

      if (global_new >= 0)
        global_new += offset_global;

      auto procs = sharing_processes->links(i);
      for (int k = 0; k < procs.rows(); ++k)
      {
        if (procs[k] == neighbours[p])
        {
          dsend[p].push_back(global_old);
          dsend[p].push_back(global_new);
        }
      }
    }

    // std::cout << "Calling isend" << std::endl;
    MPI_Isend(dsend[p].data(), dsend[p].size(), MPI_INT, p, 0, comm,
              &requests[p]);
  }

  // Receive (global old, global new) pairs to neighbours
  std::vector<std::vector<int>> drecv(num_neighbours);
  for (int p = 0; p < num_neighbours; ++p)
  {
    auto it = num_receive.find(neighbours[p]);
    assert(it != num_receive.end());
    drecv[p].resize(it->second);
    MPI_Irecv(drecv[p].data(), drecv[p].size(), MPI_INT, p, 0, comm,
              &requests[num_neighbours + p]);
  }

  // Wait for communication to finish
  std::vector<MPI_Status> status(2 * num_neighbours);
  MPI_Waitall(requests.size(), requests.data(), status.data());

  // Unpack received (global old, global new) pairs
  std::map<std::int64_t, std::int64_t> global_old_new;
  for (std::size_t i = 0; i < drecv.size(); ++i)
  {
    for (std::size_t j = 0; j < drecv[i].size(); j += 2)
      if (drecv[i][j + 1] >= 0)
        global_old_new.insert({drecv[i][j], drecv[i][j + 1]});
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
std::pair<graph::AdjacencyList<std::int32_t>,
          std::map<std::int64_t, std::int32_t>>
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

  // FIXME: Update AdjacencyList to avoid this
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& offsets
      = cells.offsets();
  std::vector<std::int32_t> _offsets(offsets.data(),
                                     offsets.data() + offsets.rows());
  return {graph::AdjacencyList<std::int32_t>(array_local, _offsets),
          global_to_local};
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int32_t>, common::IndexMap,
           std::vector<std::int64_t>>
PartitioningNew::create_distributed_adjacency_list(
    MPI_Comm comm, const mesh::Topology& topology_local,
    const std::map<std::int64_t, std::int32_t>& global_to_local_vertices)
{
  const int dim = topology_local.dim();

  // Get maximum global index across all processes
  std::int64_t my_max_global_index = 0;
  if (!global_to_local_vertices.empty())
    my_max_global_index = global_to_local_vertices.rbegin()->first;
  const std::int64_t max_global_index
      = dolfinx::MPI::all_reduce(comm, my_max_global_index, MPI_MAX);

  // Get marker for each vertex indicating if it interior or on the
  // boundary of the local topology
  const std::vector<bool>& exterior_vertex
      = compute_vertex_exterior_markers(topology_local);

  // Compute number of possibly shared vertices to send to each process
  const int size = dolfinx::MPI::size(comm);
  std::vector<int> number_to_send(size, 0);
  for (auto vertex : global_to_local_vertices)
  {
    // Only consider vertices that are the boundary of the local
    // topology
    if (exterior_vertex[vertex.second])
    {
      const int owner
          = dolfinx::MPI::index_owner(comm, vertex.first, max_global_index + 1);
      number_to_send[owner] += 1;
    }
  }

  // Compute global vertex send displacements
  std::vector<int> disp_send(size + 1, 0);
  std::partial_sum(number_to_send.begin(), number_to_send.end(),
                   disp_send.begin() + 1);

  // std::cout << "Stage 4" << std::endl;

  // Pack global vertex send data
  std::vector<std::int64_t> vertices_send(disp_send.back());
  std::vector<int> disp_tmp = disp_send;
  for (auto vertex : global_to_local_vertices)
  {
    if (exterior_vertex[vertex.second])
    {
      const int owner
          = dolfinx::MPI::index_owner(comm, vertex.first, max_global_index + 1);
      vertices_send[disp_tmp[owner]++] = vertex.first;
    }
  }

  // std::cout << "Stage 5" << std::endl;

  // Send/receive number of vertex indices to communicate to each process
  std::vector<int> number_to_recv(size);
  MPI_Alltoall(number_to_send.data(), 1, MPI_INT, number_to_recv.data(), 1,
               MPI_INT, comm);

  // std::cout << "Stage 6" << std::endl;

  // Compute receive displacements
  std::vector<int> disp_recv(size + 1, 0);
  std::partial_sum(number_to_recv.begin(), number_to_recv.end(),
                   disp_recv.begin() + 1);

  // std::cout << "Stage 7" << std::endl;

  // Send/receive global indices
  std::vector<std::int64_t> vertices_recv(disp_recv.back());
  MPI_Alltoallv(vertices_send.data(), number_to_send.data(), disp_send.data(),
                MPI::mpi_type<std::int64_t>(), vertices_recv.data(),
                number_to_recv.data(), disp_recv.data(),
                MPI::mpi_type<std::int64_t>(), comm);

  // std::cout << "Stage 8" << std::endl;

  // Build list of sharing processes for each vertex
  const std::array<std::int64_t, 2> range
      = dolfinx::MPI::local_range(comm, max_global_index + 1);
  std::vector<std::set<int>> owners(range[1] - range[0]);
  for (int i = 0; i < size; ++i)
  {
    assert((i + 1) < (int)disp_recv.size());
    for (int j = disp_recv[i]; j < disp_recv[i + 1]; ++j)
    {
      // Get back to zero reference index
      assert(j < (int)vertices_recv.size());
      const std::int64_t index = vertices_recv[j] - range[0];
      assert(index < (int)owners.size());
      owners[index].insert(i);
    }
  }

  // std::cout << "Stage 9" << std::endl;

  // For each vertex, build list of sharing processes
  std::map<std::int64_t, std::set<int>> global_vertex_to_procs;
  for (int i = 0; i < size; ++i)
  {
    for (int j = disp_recv[i]; j < disp_recv[i + 1]; ++j)
      global_vertex_to_procs[vertices_recv[j]].insert(i);
  }

  // std::cout << "Stage 10" << std::endl;

  // Get list of sharing process for each vertex
  std::unique_ptr<const graph::AdjacencyList<int>> sharing_processes;
  {
    // Pack process that share each vertex
    std::vector<int> data_send, disp_send(size + 1, 0), num_send(size);
    for (int i = 0; i < size; ++i)
    {
      for (int j = disp_recv[i]; j < disp_recv[i + 1]; ++j)
      {
        const std::int64_t vertex = vertices_recv[j];
        auto it = global_vertex_to_procs.find(vertex);
        assert(it != global_vertex_to_procs.end());
        data_send.push_back(it->second.size());
        data_send.insert(data_send.end(), it->second.begin(), it->second.end());
      }
      disp_send[i + 1] = data_send.size();
      num_send[i] = disp_send[i + 1] - disp_send[i];
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
      for (int i = disp_recv[p]; i < disp_recv[p + 1]; ++i)
      {
        const int num_procs = data_recv[i];
        const int pos = i;
        for (int j = 1; j <= num_procs; ++j)
        {
          processes.push_back(data_recv[pos + j]);
          i += 1;
        }
        process_offsets.push_back(process_offsets.back() + num_procs);
      }
    }

    sharing_processes = std::make_unique<const graph::AdjacencyList<int>>(
        processes, process_offsets);
  }

  // std::cout << "Stage 11" << std::endl;

  const int rank = dolfinx::MPI::rank(comm);
  // if (rank == 0)
  // {
  //   std::cout << "--------------" << std::endl;
  //   for (int i = 0; i < sharing_processes->num_nodes(); ++i)
  //   {
  //     // std::cout << "Vertex: " << i << ", " << vertices_send[i] <<
  //     std::endl; auto p = sharing_processes->links(i);
  //     // std::cout << "  ProcsData: " << p << std::endl;
  //   }
  // }

  std::map<std::int64_t, std::int32_t> global_to_local_owned0;
  for (auto& vertex : global_to_local_vertices)
  {
    if (!exterior_vertex[vertex.second])
      global_to_local_owned0.insert(vertex);
  }

  std::map<std::int64_t, std::int32_t> global_to_local_owned1,
      global_to_local_unowned;
  for (int i = 0; i < sharing_processes->num_nodes(); ++i)
  {
    auto it = global_to_local_vertices.find(vertices_send[i]);
    assert(it != global_to_local_vertices.end());
    if (sharing_processes->num_links(i) == 1)
      global_to_local_owned0.insert(*it);
    else if (sharing_processes->links(i).minCoeff() == rank)
      global_to_local_owned1.insert(*it);
    else
      global_to_local_unowned.insert(*it);
  }

  // Re-number owned vertices
  auto map_vertex = topology_local.index_map(0);
  assert(map_vertex);
  std::vector<std::int64_t> local_to_original;
  std::vector<std::int32_t> local_to_local_new(map_vertex->size_local(), -1);
  std::int32_t p = 0;
  for (auto it = global_to_local_owned0.begin();
       it != global_to_local_owned0.end(); ++it)
  {
    assert(it->second < (int)local_to_local_new.size());
    local_to_original.push_back(it->first);
    local_to_local_new[it->second] = p++;
  }
  for (auto it = global_to_local_owned1.begin();
       it != global_to_local_owned1.end(); ++it)
  {
    assert(it->second < (int)local_to_local_new.size());
    local_to_original.push_back(it->first);
    local_to_local_new[it->second] = p++;
  }

  // Compute process offset
  const std::int64_t num_owned_vertices
      = global_to_local_owned0.size() + global_to_local_owned1.size();
  const std::int64_t offset_global
      = dolfinx::MPI::global_offset(comm, num_owned_vertices, true);
  // std::cout << "My offset: " << rank << ", " << offset_global << std::endl;

  // Send global new global numbers
  for (int i = 0; i < sharing_processes->num_nodes(); ++i)
  {
    // Get global-local
    auto it = global_to_local_vertices.find(vertices_send[i]);
    // const std::int64_t

    assert(it != global_to_local_vertices.end());
    if (sharing_processes->num_links(i) == 1)
      global_to_local_owned0.insert(*it);
    else if (sharing_processes->links(i).minCoeff() == rank)
      global_to_local_owned1.insert(*it);
  }

  const Eigen::Array<int, Eigen::Dynamic, 1>& procs
      = sharing_processes->array();
  std::vector<int> neighbours(procs.data(), procs.data() + procs.rows());
  std::sort(neighbours.begin(), neighbours.end());
  neighbours.erase(std::unique(neighbours.begin(), neighbours.end()),
                   neighbours.end());

  std::map<int, int> num_receive;
  for (auto p : neighbours)
  {
    num_receive[p]
        = 2 * std::count(procs.data(), procs.data() + procs.rows(), p);
  }

  const int num_neighbours = neighbours.size();
  std::vector<MPI_Request> requests(2 * num_neighbours);
  std::vector<std::vector<int>> dsend(neighbours.size());
  for (std::size_t p = 0; p < neighbours.size(); ++p)
  {
    // std::vector<int> dsend;
    for (int i = 0; i < sharing_processes->num_nodes(); ++i)
    {
      auto it = global_to_local_vertices.find(vertices_send[i]);

      assert(it != global_to_local_vertices.end());
      const std::int64_t global_old = it->first;
      const std::int32_t local_old = it->second;
      std::int64_t global_new = local_to_local_new[local_old];

      if (global_new >= 0)
        global_new += offset_global;

      auto procs = sharing_processes->links(i);
      for (int k = 0; k < procs.rows(); ++k)
      {
        if (procs[k] == neighbours[p])
        {
          dsend[p].push_back(global_old);
          dsend[p].push_back(global_new);
        }
      }
    }

    // std::cout << "Calling isend" << std::endl;
    MPI_Isend(dsend[p].data(), dsend[p].size(), MPI_INT, p, 0, comm,
              &requests[p]);
  }

  std::vector<std::vector<int>> drecv(neighbours.size());
  for (std::size_t p = 0; p < neighbours.size(); ++p)
  {
    auto it = num_receive.find(neighbours[p]);
    assert(it != num_receive.end());

    drecv[p].resize(it->second);
    MPI_Irecv(drecv[p].data(), drecv[p].size(), MPI_INT, p, 0, comm,
              &requests[num_neighbours + p]);
  }

  std::vector<MPI_Status> status(2 * num_neighbours);
  MPI_Waitall(requests.size(), requests.data(), status.data());

  std::map<std::int64_t, std::int64_t> global_old_new;
  for (std::size_t i = 0; i < drecv.size(); ++i)
  {
    for (std::size_t j = 0; j < drecv[i].size(); j += 2)
      if (drecv[i][j + 1] >= 0)
        global_old_new.insert({drecv[i][j], drecv[i][j + 1]});
  }

  // Add ghosts

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

  return {graph::AdjacencyList<std::int32_t>(data_new, _offsets),
          common::IndexMap(comm, num_owned_vertices, ghosts, 1),
          std::move(local_to_original)};
}
//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int64_t>, std::vector<int>>
PartitioningNew::distribute(const MPI_Comm& comm,
                            const graph::AdjacencyList<std::int64_t>& list,
                            const std::vector<int>& owner)
{
  const int size = dolfinx::MPI::size(comm);

  // Compute number of links to send to each process
  std::vector<int> num_per_dest_send(size, 0);
  assert(list.num_nodes() == (int)owner.size());
  for (int i = 0; i < list.num_nodes(); ++i)
    num_per_dest_send[owner[i]] += list.num_links(i) + 1;

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
  std::vector<std::int64_t> data_send(list.array().rows() + list.num_nodes());
  for (int i = 0; i < list.num_nodes(); ++i)
  {
    const int dest = owner[i];
    auto links = list.links(i);
    data_send[offset[dest]] = links.rows();
    ++offset[dest];
    for (int j = 0; j < links.rows(); ++j)
      data_send[offset[dest] + j] = links(j);
    offset[dest] += links.rows();
  }

  // Send/receive data
  std::vector<std::int64_t> data_recv(disp_recv.back());
  MPI_Alltoallv(data_send.data(), num_per_dest_send.data(), disp_send.data(),
                MPI_INT64_T, data_recv.data(), num_per_dest_recv.data(),
                disp_recv.data(), MPI_INT64_T, comm);

  // Unpack receive buffer
  std::vector<std::int64_t> array;
  std::vector<std::int32_t> list_offset(1, 0);
  std::vector<int> src;
  for (std::size_t p = 0; p < disp_recv.size() - 1; ++p)
  {
    for (int i = disp_recv[p]; i < disp_recv[p + 1]; ++i)
    {
      src.push_back(p);
      const std::int64_t num_links = data_recv[i];
      const int pos = i;
      for (int j = 1; j <= num_links; ++j)
      {
        array.push_back(data_recv[pos + j]);
        i += 1;
      }
      list_offset.push_back(list_offset.back() + num_links);
    }
  }

  return {graph::AdjacencyList<std::int64_t>(array, list_offset), src};
}
//-----------------------------------------------------------------------------
