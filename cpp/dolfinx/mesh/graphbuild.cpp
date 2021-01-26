// Copyright (C) 2010-2021 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "graphbuild.h"
#include <algorithm>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/cell_types.h>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{

//-----------------------------------------------------------------------------
// Compute local part of the dual graph, and return return (local_graph,
// facet_cell_map, number of local edges in the graph (undirected)
template <int N>
std::pair<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>>
compute_local_dual_graph_keyed(
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type)
{
  common::Timer timer("Compute local part of mesh dual graph");

  const int tdim = mesh::cell_dim(cell_type);
  const std::int32_t num_local_cells = cell_vertices.num_nodes();
  const int num_facets_per_cell = mesh::cell_num_entities(cell_type, tdim - 1);
  const int num_vertices_per_facet
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  assert(N == num_vertices_per_facet);

  // Compute edges (cell-cell connections) using local numbering

  // Create map from cell vertices to entity vertices
  auto facet_vertices = mesh::get_entity_vertices(cell_type, tdim - 1);

  // Vector-of-arrays data structure, which is considerably faster than
  // vector-of-vectors
  std::vector<std::pair<std::array<std::int64_t, N>, std::int32_t>> facets(
      num_facets_per_cell * num_local_cells);

  // Iterate over all cells and build list of all facets (keyed on
  // sorted vertex indices), with cell index attached
  int counter = 0;
  for (std::int32_t i = 0; i < num_local_cells; ++i)
  {
    // Iterate over facets of cell
    auto vertices = cell_vertices.links(i);
    for (int j = 0; j < num_facets_per_cell; ++j)
    {
      // Get list of facet vertices
      std::array<std::int64_t, N>& facet = facets[counter].first;
      for (int k = 0; k < N; ++k)
        facet[k] = vertices[facet_vertices(j, k)];

      // Sort facet vertices
      std::sort(facet.begin(), facet.end());

      // Attach local cell index
      facets[counter].second = i;

      // Increment facet counter
      counter++;
    }
  }

  // Sort facets
  std::sort(facets.begin(), facets.end());

  // Find maching facets by comparing facet i and facet i -1
  std::vector<std::int32_t> num_local_graph(num_local_cells, 0);
  std::vector<std::int64_t> facet_cell_map;
  std::vector<bool> facet_match(facets.size(), false);
  bool this_equal, last_equal = false;
  for (std::size_t i = 1; i < facets.size(); ++i)
  {
    const auto& facet0 = facets[i - 1].first;
    const auto& facet1 = facets[i].first;
    this_equal = std::equal(facet0.begin(), facet0.end(), facet1.begin());
    if (this_equal)
    {
      if (last_equal)
      {
        LOG(ERROR) << "Found three identical facets in mesh (local)";
        throw std::runtime_error(
            "Inconsistent mesh data in GraphBuilder: three identical facets");
      }

      // Add edges (directed graph, so add both ways)
      const int cell_index0 = facets[i - 1].second;
      const int cell_index1 = facets[i].second;
      num_local_graph[cell_index0] += 1;
      num_local_graph[cell_index1] += 1;

      facet_match[i] = true;
    }
    else if (!this_equal and !last_equal)
    {
      // No match, so add facet0 to map
      const int cell_index0 = facets[i - 1].second;
      facet_cell_map.insert(facet_cell_map.end(), facet0.begin(), facet0.end());
      facet_cell_map.push_back(cell_index0);
    }

    last_equal = this_equal;
  }

  // Add last facet, as it's not covered by the above loop
  if (!facets.empty() and !last_equal)
  {
    const int k = facets.size() - 1;
    const int cell_index = facets[k].second;
    facet_cell_map.insert(facet_cell_map.end(), facets[k].first.begin(),
                          facets[k].first.end());
    facet_cell_map.push_back(cell_index);
  }

  // Build adjacency list data
  std::vector<std::int32_t> offsets(num_local_graph.size() + 1, 0);
  std::partial_sum(num_local_graph.begin(), num_local_graph.end(),
                   std::next(offsets.begin(), 1));
  std::vector<std::int32_t> local_graph_data(offsets.back());
  std::vector<int> pos(num_local_cells, 0);
  for (std::size_t i = 1; i < facets.size(); ++i)
  {
    if (facet_match[i])
    {
      // Add edges (directed graph, so add both ways)
      const int cell_index0 = facets[i - 1].second;
      const int cell_index1 = facets[i].second;
      local_graph_data[offsets[cell_index0] + pos[cell_index0]++] = cell_index1;
      local_graph_data[offsets[cell_index1] + pos[cell_index1]++] = cell_index0;
    }
  }

  return {graph::AdjacencyList<std::int32_t>(std::move(local_graph_data),
                                             std::move(offsets)),
          std::move(facet_cell_map)};
}
//-----------------------------------------------------------------------------
// Build nonlocal part of dual graph for mesh and return number of
// non-local edges. Note: GraphBuilder::compute_local_dual_graph should
// be called before this function is called. Returns (ghost vertices,
// num_nonlocal_edges)
std::pair<graph::AdjacencyList<std::int64_t>, std::int32_t>
compute_nonlocal_dual_graph(
    const MPI_Comm comm,
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type,
    const std::vector<std::int64_t>& facet_cell_map,
    const graph::AdjacencyList<std::int32_t>& local_graph)
{
  LOG(INFO) << "Build nonlocal part of mesh dual graph";
  common::Timer timer("Compute non-local part of mesh dual graph");

  // Get number of MPI processes, and return if mesh is not distributed
  const int num_processes = dolfinx::MPI::size(comm);
  if (num_processes == 1)
  {
    // Convert graph to int64
    return {graph::AdjacencyList<std::int64_t>(
                std::vector<std::int64_t>(local_graph.array().begin(),
                                          local_graph.array().end()),
                local_graph.offsets()),
            0};
  }

  const int tdim = mesh::cell_dim(cell_type);
  const int num_vertices_per_facet
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  // At this stage facet_cell map only contains facets->cells with edge
  // facets either interprocess or external boundaries

  // Find the global range of the first vertex index of each facet in the list
  // and use this to divide up the facets between all processes.

  // TODO: improve scalability, possibly by limiting the number of
  // processes which do the matching, and using a neighbor comm?
  std::int64_t local_min = std::numeric_limits<std::int64_t>::max();
  std::int64_t local_max = 0;
  assert(facet_cell_map.size() % (num_vertices_per_facet + 1) == 0);
  for (std::size_t i = 0; i < facet_cell_map.size();
       i += num_vertices_per_facet + 1)
  {
    local_min = std::min(local_min, facet_cell_map[i]);
    local_max = std::max(local_max, facet_cell_map[i]);
  }

  std::int64_t global_min, global_max;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT64_T, MPI_MIN, comm);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT64_T, MPI_MAX, comm);
  const std::int64_t global_range = global_max - global_min + 1;

  // Send facet-cell map to intermediary match-making processes

  // Get cell offset for this process to create global numbering for cells
  const std::int32_t num_local_cells = cell_vertices.num_nodes();
  const std::int64_t cell_offset
      = dolfinx::MPI::global_offset(comm, num_local_cells, true);

  // Count number of item to send to each rank
  std::vector<int> p_count(num_processes, 0);
  for (std::size_t i = 0; i < facet_cell_map.size();
       i += num_vertices_per_facet + 1)
  {
    // Use first vertex of facet to partition into blocks
    const int dest_proc = dolfinx::MPI::index_owner(
        num_processes, facet_cell_map[i] - global_min, global_range);
    p_count[dest_proc] += num_vertices_per_facet + 1;
  }

  // Create back adjacency list send buffer
  std::vector<std::int32_t> offsets(num_processes + 1, 0);
  std::partial_sum(p_count.begin(), p_count.end(),
                   std::next(offsets.begin(), 1));
  graph::AdjacencyList<std::int64_t> send_buffer(
      std::vector<std::int64_t>(offsets.back()), std::move(offsets));

  // Pack map data and send to match-maker process
  std::vector<int> pos(send_buffer.num_nodes(), 0);
  for (std::size_t i = 0; i < facet_cell_map.size();
       i += num_vertices_per_facet + 1)
  {
    tcb::span<const std::int64_t> facet(&facet_cell_map[i],
                                        num_vertices_per_facet + 1);
    const int dest_proc = dolfinx::MPI::index_owner(
        num_processes, facet[0] - global_min, global_range);
    tcb::span<std::int64_t> buffer = send_buffer.links(dest_proc);
    std::copy(facet.begin(), facet.end(),
              std::next(buffer.begin(), pos[dest_proc]));
    buffer[pos[dest_proc] + num_vertices_per_facet] += cell_offset;
    pos[dest_proc] += facet.size();
  }

  // Send data
  graph::AdjacencyList<std::int64_t> recvd_buffer
      = dolfinx::MPI::all_to_all(comm, send_buffer);
  assert(recvd_buffer.array().size() % (num_vertices_per_facet + 1) == 0);
  const int num_facets
      = recvd_buffer.array().size() / (num_vertices_per_facet + 1);

  // Set up vector of owning processes for each received facet
  const std::vector<std::int32_t>& recvd_buffer_offsets
      = recvd_buffer.offsets();
  std::vector<int> proc(num_facets);
  for (int p = 0; p < num_processes; ++p)
  {
    for (int j = recvd_buffer_offsets[p] / (num_vertices_per_facet + 1);
         j < recvd_buffer_offsets[p + 1] / (num_vertices_per_facet + 1); ++j)
    {
      proc[j] = p;
    }
  }

  // Reshape the return buffer
  {
    std::vector<std::int32_t> offsets(num_facets + 1, 0);
    for (std::size_t i = 0; i < offsets.size() - 1; ++i)
      offsets[i + 1] = offsets[i] + (num_vertices_per_facet + 1);
    recvd_buffer = graph::AdjacencyList<std::int64_t>(
        std::move(recvd_buffer.array()), std::move(offsets));
  }

  // Get permutation that takes facets into sorted order
  std::vector<int> perm(num_facets);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&recvd_buffer](int a, int b) {
    return std::lexicographical_compare(
        recvd_buffer.links(a).begin(), std::prev(recvd_buffer.links(a).end()),
        recvd_buffer.links(b).begin(), std::prev(recvd_buffer.links(b).end()));
  });

  // Count data items to send to each rank
  p_count.assign(num_processes, 0);
  bool this_equal, last_equal = false;
  std::vector<bool> facet_match(num_facets, false);
  for (int i = 1; i < num_facets; ++i)
  {
    const int i0 = perm[i - 1];
    const int i1 = perm[i];
    const auto facet0 = recvd_buffer.links(i0);
    const auto facet1 = recvd_buffer.links(i1);
    this_equal
        = std::equal(facet0.begin(), std::prev(facet0.end()), facet1.begin());
    if (this_equal)
    {
      if (last_equal)
      {
        LOG(ERROR) << "Found three identical facets in mesh (match process)";
        throw std::runtime_error("Inconsistent mesh data in GraphBuilder: "
                                 "found three identical facets");
      }
      p_count[proc[i0]] += 2;
      p_count[proc[i1]] += 2;
      facet_match[i] = true;
    }
    last_equal = this_equal;
  }

  // Create back adjacency list send buffer
  offsets.assign(num_processes + 1, 0);
  std::partial_sum(p_count.begin(), p_count.end(),
                   std::next(offsets.begin(), 1));
  send_buffer = graph::AdjacencyList<std::int64_t>(
      std::vector<std::int64_t>(offsets.back()), std::move(offsets));

  pos.assign(send_buffer.num_nodes(), 0);
  for (int i = 1; i < num_facets; ++i)
  {
    if (facet_match[i])
    {
      const int i0 = perm[i - 1];
      const int i1 = perm[i];
      const int proc0 = proc[i0];
      const int proc1 = proc[i1];
      const auto facet0 = recvd_buffer.links(i0);
      const auto facet1 = recvd_buffer.links(i1);

      const std::int64_t cell0 = facet0[num_vertices_per_facet];
      const std::int64_t cell1 = facet1[num_vertices_per_facet];

      auto buffer0 = send_buffer.links(proc0);
      buffer0[pos[proc0]++] = cell0;
      buffer0[pos[proc0]++] = cell1;
      auto buffer1 = send_buffer.links(proc1);
      buffer1[pos[proc1]++] = cell1;
      buffer1[pos[proc1]++] = cell0;
    }
  }

  // Send matches to other processes
  const std::vector<std::int64_t> cell_list
      = dolfinx::MPI::all_to_all(comm, send_buffer).array();

  // Ghost nodes: insert connected cells into local map

  // Count number of adjacency list edges
  std::vector<int> edge_count(local_graph.num_nodes(), 0);
  for (int i = 0; i < local_graph.num_nodes(); ++i)
    edge_count[i] += local_graph.num_links(i);
  for (std::size_t i = 0; i < cell_list.size(); i += 2)
  {
    assert(cell_list[i] - cell_offset >= 0);
    assert(cell_list[i] - cell_offset < (std::int64_t)edge_count.size());
    edge_count[cell_list[i] - cell_offset] += 1;
  }

  // Build adjacency list
  offsets.assign(edge_count.size() + 1, 0);
  std::partial_sum(edge_count.begin(), edge_count.end(),
                   std::next(offsets.begin(), 1));
  graph::AdjacencyList<std::int64_t> graph(
      std::vector<std::int64_t>(offsets.back()), std::move(offsets));
  pos.assign(graph.num_nodes(), 0);
  std::vector<std::int64_t> ghost_nodes;
  for (int i = 0; i < local_graph.num_nodes(); ++i)
  {
    auto local_graph_i = local_graph.links(i);
    auto graph_i = graph.links(i);
    for (std::size_t j = 0; j < local_graph_i.size(); ++j)
      graph_i[pos[i]++] = local_graph_i[j] + cell_offset;
  }

  for (std::size_t i = 0; i < cell_list.size(); i += 2)
  {
    const std::size_t node = cell_list[i] - cell_offset;
    auto edges = graph.links(node);
#ifdef DEBUG
    if (auto it_end = std::next(edges.begin(), pos[node]);
        std::find(edges.begin(), it_end, cell_list[i + 1]) != it_end)
    {
      LOG(ERROR) << "Received same edge twice in dual graph";
      throw std::runtime_error("Inconsistent mesh data in GraphBuilder: "
                               "received same edge twice in dual graph");
    }
#endif
    edges[pos[node]++] = cell_list[i + 1];
    ghost_nodes.push_back(cell_list[i + 1]);
  }

  std::sort(ghost_nodes.begin(), ghost_nodes.end());
  const std::int32_t num_ghost_nodes = std::distance(
      ghost_nodes.begin(), std::unique(ghost_nodes.begin(), ghost_nodes.end()));

  return {std::move(graph), num_ghost_nodes};
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int64_t>, std::array<std::int32_t, 2>>
mesh::build_dual_graph(const MPI_Comm mpi_comm,
                       const graph::AdjacencyList<std::int64_t>& cell_vertices,
                       const mesh::CellType& cell_type)
{
  LOG(INFO) << "Build mesh dual graph";

  // Compute local part of dual graph
  auto [local_graph, facet_cell_map]
      = mesh::build_local_dual_graph(cell_vertices, cell_type);

  // Compute nonlocal part
  auto [graph, num_ghost_nodes] = compute_nonlocal_dual_graph(
      mpi_comm, cell_vertices, cell_type, facet_cell_map, local_graph);

  LOG(INFO) << "Graph edges (local:" << local_graph.offsets().back()
            << ", non-local:"
            << graph.offsets().back() - local_graph.offsets().back() << ")";

  return {std::move(graph), {num_ghost_nodes, local_graph.offsets().back()}};
}
//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>>
mesh::build_local_dual_graph(
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type)
{
  LOG(INFO) << "Build local part of mesh dual graph";

  const int tdim = mesh::cell_dim(cell_type);
  const int num_entity_vertices
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  switch (num_entity_vertices)
  {
  case 1:
    return compute_local_dual_graph_keyed<1>(cell_vertices, cell_type);
  case 2:
    return compute_local_dual_graph_keyed<2>(cell_vertices, cell_type);
  case 3:
    return compute_local_dual_graph_keyed<3>(cell_vertices, cell_type);
  case 4:
    return compute_local_dual_graph_keyed<4>(cell_vertices, cell_type);
  default:
    throw std::runtime_error(
        "Cannot compute local part of dual graph. Entities with "
        + std::to_string(num_entity_vertices) + " vertices not supported");
  }
}
//-----------------------------------------------------------------------------
