// Copyright (C) 2010-2013 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "GraphBuilder.h"
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/cell_types.h>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{

//-----------------------------------------------------------------------------
// Compute local part of the dual graph, and return return (local_graph,
// facet_cell_map, number of local edges in the graph (undirected)
template <int N>
std::tuple<std::vector<std::vector<std::int32_t>>,
           std::vector<std::pair<std::vector<std::int64_t>, std::int32_t>>,
           std::int32_t>
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
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      facet_vertices = mesh::get_entity_vertices(cell_type, tdim - 1);

  // Vector-of-arrays data structure, which is considerably faster than
  // vector-of-vectors
  std::vector<std::pair<std::array<std::int64_t, N>, std::int32_t>> facets(
      num_facets_per_cell * num_local_cells);

  // Iterate over all cells and build list of all facets (keyed on
  // sorted vertex indices), with cell index attached
  int counter = 0;
  for (std::int32_t i = 0; i < num_local_cells; ++i)
  {
    auto& vertices = cell_vertices.links(i);
    // Iterate over facets of cell
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
  std::size_t num_local_edges = 0;
  std::vector<std::vector<std::int32_t>> local_graph(num_local_cells);
  std::vector<std::pair<std::vector<std::int64_t>, std::int32_t>>
      facet_cell_map;
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
        LOG(ERROR) << "Found three identical facets in mesh";
        throw std::runtime_error("fail");
      }

      // Add edges (directed graph, so add both ways)
      const int cell_index0 = facets[i - 1].second;
      const int cell_index1 = facets[i].second;
      local_graph[cell_index0].push_back(cell_index1);
      local_graph[cell_index1].push_back(cell_index0);
      // Increment number of local edges found
      ++num_local_edges;
    }
    else if (!this_equal and !last_equal)
    {
      // No match, so add facet0 to map
      const int cell_index0 = facets[i - 1].second;
      facet_cell_map.emplace_back(
          std::vector<std::int64_t>(facet0.begin(), facet0.end()), cell_index0);
    }

    last_equal = this_equal;
  }

  // Add last facet, as it's not covered by the above loop.
  if (!facets.empty() and !last_equal)
  {
    const int k = facets.size() - 1;
    const int cell_index = facets[k].second;
    facet_cell_map.emplace_back(
        std::vector<std::int64_t>(facets[k].first.begin(),
                                  facets[k].first.end()),
        cell_index);
  }

  return {std::move(local_graph), std::move(facet_cell_map), num_local_edges};
} // namespace
//-----------------------------------------------------------------------------
// Build nonlocal part of dual graph for mesh and return number of
// non-local edges. Note: GraphBuilder::compute_local_dual_graph should
// be called before this function is called. Returns (ghost vertices,
// num_nonlocal_edges)
std::tuple<std::vector<std::vector<std::int64_t>>, std::int32_t, std::int32_t>
compute_nonlocal_dual_graph(
    const MPI_Comm mpi_comm,
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type,
    const std::vector<std::pair<std::vector<std::int64_t>, std::int32_t>>&
        facet_cell_map,
    const std::vector<std::vector<std::int32_t>>& local_graph)
{
  LOG(INFO) << "Build nonlocal part of mesh dual graph";
  common::Timer timer("Compute non-local part of mesh dual graph");

  // Get cell offset for this process and add to local graph
  const std::int32_t num_local_cells = cell_vertices.num_nodes();
  const std::int64_t offset
      = dolfinx::MPI::global_offset(mpi_comm, num_local_cells, true);

  std::vector<std::vector<std::int64_t>> graph(local_graph.size());
  for (std::size_t i = 0; i < local_graph.size(); ++i)
  {
    graph[i] = std::vector<std::int64_t>(local_graph[i].begin(),
                                         local_graph[i].end());
    std::for_each(graph[i].begin(), graph[i].end(),
                  [offset](auto& n) { n += offset; });
  }

  // Get number of MPI processes, and return if mesh is not distributed
  const int num_processes = dolfinx::MPI::size(mpi_comm);
  if (num_processes == 1)
    return {graph, 0, 0};

  // At this stage facet_cell map only contains facets->cells with edge
  // facets either interprocess or external boundaries

  // List of cell vertices
  const int tdim = mesh::cell_dim(cell_type);
  const int num_vertices_per_facet
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  // Compute local edges (cell-cell connections) using global (internal
  // to this function, not the user numbering) numbering

  // Find the global range of the first vertex index of each facet in the list
  // and use this to divide up the facets between all processes.
  // TODO: improve scalability, possibly by limiting the number of processes
  // which do the matching, and using a neighbor comm?
  std::int64_t local_min = std::numeric_limits<std::int64_t>::max();
  std::int64_t local_max = 0;
  for (const auto& it : facet_cell_map)
  {
    const std::vector<std::int64_t>& facet = it.first;
    local_min = std::min(local_min, facet[0]);
    local_max = std::max(local_max, facet[0]);
  }

  std::int64_t global_min, global_max;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT64_T, MPI_MIN, mpi_comm);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT64_T, MPI_MAX, mpi_comm);
  const std::int64_t global_range = global_max - global_min + 1;

  // Send facet-cell map to intermediary match-making processes
  std::vector<std::vector<std::int64_t>> send_buffer(num_processes);

  // Pack map data and send to match-maker process
  for (const auto& it : facet_cell_map)
  {
    // FIXME: Could use a better index? First vertex is slightly
    //        skewed towards low values - may not be important

    // Use first vertex of facet to partition into blocks
    const int dest_proc = dolfinx::MPI::index_owner(
        num_processes, (it.first)[0] - global_min, global_range);

    // Pack map into vectors to send
    send_buffer[dest_proc].insert(send_buffer[dest_proc].end(),
                                  it.first.begin(), it.first.end());

    // Add offset to cell numbers sent off process
    send_buffer[dest_proc].push_back(it.second + offset);
  }

  // Send data
  const graph::AdjacencyList<std::int64_t> received_buffer
      = dolfinx::MPI::all_to_all(
          mpi_comm, graph::AdjacencyList<std::int64_t>(send_buffer));

  assert(received_buffer.array().size() % (num_vertices_per_facet + 1) == 0);
  const int num_facets
      = received_buffer.array().size() / (num_vertices_per_facet + 1);

  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
      received_buffer_array(received_buffer.array().data(), num_facets,
                            num_vertices_per_facet + 1);

  // Set up vector of owning processes for each received facet
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& received_buffer_offsets
      = received_buffer.offsets();
  std::vector<int> proc(num_facets);
  for (int p = 0; p < num_processes; ++p)
  {
    for (int j = received_buffer_offsets[p] / (num_vertices_per_facet + 1);
         j < received_buffer_offsets[p + 1] / (num_vertices_per_facet + 1); ++j)
      proc[j] = p;
  }

  // Get permutation that takes facets into sorted order
  std::vector<int> perm(num_facets);
  std::iota(perm.begin(), perm.end(), 0);
  // Lambda with capture for sort comparison
  const auto cmp
      = [&received_buffer_array, num_vertices_per_facet](int a, int b) {
          const auto facet_a = received_buffer_array.row(a);
          const auto facet_b = received_buffer_array.row(b);
          return std::lexicographical_compare(
              facet_a.data(), facet_a.data() + num_vertices_per_facet,
              facet_b.data(), facet_b.data() + num_vertices_per_facet);
        };

  std::sort(perm.begin(), perm.end(), cmp);

  // Clear send buffer
  send_buffer = std::vector<std::vector<std::int64_t>>(num_processes);

  bool this_equal, last_equal = false;
  for (int i = 1; i < num_facets; ++i)
  {
    const int i0 = perm[i - 1];
    const int i1 = perm[i];
    const auto facet0 = received_buffer_array.row(i0);
    const auto facet1 = received_buffer_array.row(i1);
    this_equal = std::equal(
        facet0.data(), facet0.data() + num_vertices_per_facet, facet1.data());
    if (this_equal)
    {
      if (last_equal)
      {
        LOG(ERROR) << "Found three identical facets in mesh";
        throw std::runtime_error("Mesh inconsistency in dual graph");
      }
      const std::int64_t cell0 = facet0[num_vertices_per_facet];
      const std::int64_t cell1 = facet1[num_vertices_per_facet];
      const int proc0 = proc[i0];
      const int proc1 = proc[i1];
      send_buffer[proc0].push_back(cell0);
      send_buffer[proc0].push_back(cell1);
      send_buffer[proc1].push_back(cell1);
      send_buffer[proc1].push_back(cell0);
    }
    last_equal = this_equal;
  }

  // Send matches to other processes
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> cell_list
      = dolfinx::MPI::all_to_all(
            mpi_comm, graph::AdjacencyList<std::int64_t>(send_buffer))
            .array();

  // Ghost nodes: insert connected cells into local map
  std::set<std::int64_t> ghost_nodes;
  std::int32_t num_nonlocal_edges = 0;
  for (int i = 0; i < cell_list.rows(); i += 2)
  {
    assert((std::int64_t)cell_list[i] >= offset);
    assert((std::int64_t)(cell_list[i] - offset)
           < (std::int64_t)local_graph.size());

    auto& edges = graph[cell_list[i] - offset];
    auto it = std::find(edges.begin(), edges.end(), cell_list[i + 1]);
    if (it == graph[cell_list[i] - offset].end())
    {
      edges.push_back(cell_list[i + 1]);
      ++num_nonlocal_edges;
    }
    else
    {
      LOG(ERROR) << "Received same edge twice in dual graph";
      throw std::runtime_error("Mesh inconsistency building dual graph");
    }
    ghost_nodes.insert(cell_list[i + 1]);
  }

  return {std::move(graph), ghost_nodes.size(), num_nonlocal_edges};
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<std::vector<std::int64_t>>, std::array<std::int32_t, 3>>
mesh::GraphBuilder::compute_dual_graph(
    const MPI_Comm mpi_comm,
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type)
{
  LOG(INFO) << "Build mesh dual graph";

  // Compute local part of dual graph
  auto [local_graph, facet_cell_map, num_local_edges]
      = mesh::GraphBuilder::compute_local_dual_graph(cell_vertices, cell_type);

  // Compute nonlocal part
  auto [graph, num_ghost_nodes, num_nonlocal_edges]
      = compute_nonlocal_dual_graph(mpi_comm, cell_vertices, cell_type,
                                    facet_cell_map, local_graph);

  LOG(INFO) << "Graph edges (local:" << num_local_edges
            << ", non-local:" << num_nonlocal_edges << ")";

  return {std::move(graph),
          {num_ghost_nodes, num_local_edges, num_nonlocal_edges}};
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<std::vector<std::int32_t>>,
           std::vector<std::pair<std::vector<std::int64_t>, std::int32_t>>,
           std::int32_t>
dolfinx::mesh::GraphBuilder::compute_local_dual_graph(
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
