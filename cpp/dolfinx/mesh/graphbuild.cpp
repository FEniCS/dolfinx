// Copyright (C) 2010-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
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
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
// Compute local part of the dual graph, and return return (local_graph,
// facet_cell_map, number of local edges in the graph (undirected)
std::pair<graph::AdjacencyList<std::int32_t>, xt::xtensor<std::int64_t, 2>>
compute_local_dual_graph_keyed(
    const xtl::span<const std::int64_t>& cell_vertices,
    const xtl::span<const std::int32_t>& cell_offsets, int tdim)
{
  common::Timer timer("Compute local part of mesh dual graph");

  const std::int32_t num_local_cells = cell_offsets.size() - 1;
  if (num_local_cells == 0)
  {
    // Empty mesh on this process
    return {graph::AdjacencyList<std::int32_t>(0),
            xt::xtensor<std::int64_t, 2>({0, 0})};
  }

  // Count number of cells of each type, based on the number of vertices
  // in each cell, covering interval(2) through to hex(8)
  std::array<int, 9> count;
  std::fill(count.begin(), count.end(), 0);
  for (int i = 0; i < num_local_cells; ++i)
  {
    const std::size_t num_cell_vertices = cell_offsets[i + 1] - cell_offsets[i];
    assert(num_cell_vertices < count.size());
    ++count[num_cell_vertices];
  }

  int num_facets = 0;
  int num_facet_vertices = 0;

  // For each topological dimension, there is a limited set of allowed
  // cell types. In 1D, interval; 2D: tri or quad, 3D: tet, prism,
  // pyramid or hex.
  //
  // To quickly look up the facets on a given cell, create a lookup
  // table, which maps from number of cell vertices->facet vertex list.
  // This is unique for each dimension 1D (interval: 2 vertices)) 2D
  // (triangle: 3, quad: 4) 3D (tet: 4, pyramid: 5, prism: 6, hex: 8)
  std::vector<graph::AdjacencyList<int>> nv_to_facets(
      9, graph::AdjacencyList<int>(0));

  switch (tdim)
  {
  case 1:
    if (count[2] != num_local_cells)
      throw std::runtime_error("Invalid cells in 1D mesh");
    nv_to_facets[2] = mesh::get_entity_vertices(mesh::CellType::interval, 0);
    num_facets = count[2] * 2;
    num_facet_vertices = 1;
    break;
  case 2:
    if ((count[3] + count[4]) != num_local_cells)
      throw std::runtime_error("Invalid cells in 2D mesh");
    nv_to_facets[3] = mesh::get_entity_vertices(mesh::CellType::triangle, 1);
    nv_to_facets[4]
        = mesh::get_entity_vertices(mesh::CellType::quadrilateral, 1);
    num_facet_vertices = 2;
    num_facets = count[3] * 3 + count[4] * 4;
    break;
  case 3:
    if ((count[4] + count[5] + count[6] + count[8]) != num_local_cells)
      throw std::runtime_error("Invalid cells in 3D mesh");

    // If any quad facets in mesh, expand to width=4
    if (count[5] > 0 or count[6] > 0 or count[8] > 0)
      num_facet_vertices = 4;
    else
      num_facet_vertices = 3;

    num_facets = count[4] * 4 + count[5] * 5 + count[6] * 5 + count[8] * 6;
    nv_to_facets[4] = mesh::get_entity_vertices(mesh::CellType::tetrahedron, 2);
    nv_to_facets[5] = mesh::get_entity_vertices(mesh::CellType::pyramid, 2);
    nv_to_facets[6] = mesh::get_entity_vertices(mesh::CellType::prism, 2);
    nv_to_facets[8] = mesh::get_entity_vertices(mesh::CellType::hexahedron, 2);
    break;
  default:
    throw std::runtime_error("Invalid tdim");
  }

  // List of facets and associated cells
  std::vector<std::array<std::int64_t, 5>> facets(num_facets);
  int counter = 0;
  for (std::int32_t i = 0; i < num_local_cells; ++i)
  {
    // Iterate over facets of cell
    xtl::span<const std::int64_t> vertices(
        std::next(cell_vertices.begin(), cell_offsets[i]),
        cell_offsets[i + 1] - cell_offsets[i]);
    const int nv = vertices.size();
    const graph::AdjacencyList<int>& f = nv_to_facets[nv];
    const int num_facets_per_cell = f.num_nodes();
    for (int j = 0; j < num_facets_per_cell; ++j)
    {
      std::array<std::int64_t, 5>& facet = facets[counter];
      facet[4] = i; // cell counter

      // Fill last entry with max_int64: for mixed 3D, when some facets
      // may be triangle adds an extra dummy vertex which will sort to
      // last position
      facet[3] = std::numeric_limits<std::int64_t>::max();

      // Get list of facet vertices
      auto f_to_v = f.links(j);
      assert(f_to_v.size() < 5);
      for (std::size_t k = 0; k < f_to_v.size(); ++k)
        facet[k] = vertices[f_to_v[k]];

      // Sort facet vertices
      std::sort(facet.begin(), std::next(facet.begin(), f_to_v.size()));

      // Increment facet counter
      counter++;
    }
  }
  assert(counter == (int)facets.size());

  // Sort facet indices
  std::sort(facets.begin(), facets.end(),
            [num_facet_vertices](const std::array<std::int64_t, 5>& fa,
                                 const std::array<std::int64_t, 5>& fb) {
              return std::lexicographical_compare(
                  fa.begin(), fa.begin() + num_facet_vertices, fb.begin(),
                  fb.begin() + num_facet_vertices);
            });

  // Stack up cells joined by facet as pairs in local_graph, and record
  // any non-matching
  std::vector<std::int32_t> local_graph;
  std::vector<std::int32_t> unmatched_facets;
  local_graph.reserve(num_local_cells * 2);
  unmatched_facets.reserve(num_local_cells);

  int eq_count = 0;
  for (std::size_t j = 1; j < facets.size(); ++j)
  {
    if (std::equal(facets[j].begin(),
                   std::next(facets[j].begin(), num_facet_vertices),
                   facets[j - 1].begin()))
    {
      ++eq_count;
      // join cells at cell_index[j] <-> cell_index[jlast]
      local_graph.push_back(facets[j].back());
      local_graph.push_back(facets[j - 1].back());

      if (eq_count > 1)
        LOG(WARNING) << "Same facet in more than two cells";
    }
    else
    {
      if (eq_count == 0)
        unmatched_facets.push_back(j - 1);
      eq_count = 0;
    }
  }

  // save last one, if unmatched...
  if (eq_count == 0)
    unmatched_facets.push_back(facets.size() - 1);

  xt::xtensor<std::int64_t, 2> facet_cell_map(
      {unmatched_facets.size(),
       static_cast<std::size_t>(num_facet_vertices + 1)});
  for (std::size_t c = 0; c < unmatched_facets.size(); ++c)
  {
    std::int32_t j = unmatched_facets[c];
    auto facetmap = xt::row(facet_cell_map, c);
    std::copy_n(facets[j].cbegin(), num_facet_vertices, facetmap.begin());
    facetmap[num_facet_vertices] = facets[j].back();
  }

  // Get connection counts for each cell
  std::vector<std::int32_t> num_local_graph(num_local_cells, 0);
  for (std::int32_t cell : local_graph)
  {
    assert(cell < num_local_cells);
    ++num_local_graph[cell];
  }
  std::vector<std::int32_t> offsets(num_local_graph.size() + 1, 0);
  std::partial_sum(num_local_graph.begin(), num_local_graph.end(),
                   std::next(offsets.begin(), 1));
  std::vector<std::int32_t> local_graph_data(offsets.back());

  // Build adjacency data
  std::vector<std::int32_t> pos(offsets.begin(), offsets.end() - 1);
  for (std::size_t i = 0; i < local_graph.size(); i += 2)
  {
    const std::size_t c0 = local_graph[i];
    const std::size_t c1 = local_graph[i + 1];
    assert(c0 < pos.size());
    assert(c1 < pos.size());
    assert(pos[c0] < (int)local_graph_data.size());
    assert(pos[c1] < (int)local_graph_data.size());
    local_graph_data[pos[c0]++] = c1;
    local_graph_data[pos[c1]++] = c0;
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
    const MPI_Comm comm, std::int32_t num_local_cells,
    const xt::xtensor<std::int64_t, 2>& facet_cell_map,
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

  // Some processes may have empty map, so get max across all
  const std::int32_t num_vertices_local = facet_cell_map.shape(1) - 1;
  std::int32_t num_vertices_per_facet;
  MPI_Allreduce(&num_vertices_local, &num_vertices_per_facet, 1, MPI_INT32_T,
                MPI_MAX, comm);
  LOG(INFO) << "nv per facet=" << num_vertices_per_facet << "\n";

  // At this stage facet_cell map only contains facets->cells with edge
  // facets either interprocess or external boundaries

  // Find the global range of the first vertex index of each facet in the list
  // and use this to divide up the facets between all processes.

  // TODO: improve scalability, possibly by limiting the number of
  // processes which do the matching, and using a neighbor comm?
  std::int64_t local_min = std::numeric_limits<std::int64_t>::max();
  std::int64_t local_max = 0;
  if (facet_cell_map.shape(0) > 0)
  {
    std::array<std::int64_t, 2> p = xt::minmax(xt::col(facet_cell_map, 0))();
    local_min = p[0];
    local_max = p[1];
  }

  std::int64_t global_min, global_max;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT64_T, MPI_MIN, comm);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT64_T, MPI_MAX, comm);
  const std::int64_t global_range = global_max - global_min + 1;

  // Send facet-cell map to intermediary match-making processes

  // Get cell offset for this process to create global numbering for cells
  const std::int64_t cell_offset
      = dolfinx::MPI::global_offset(comm, num_local_cells, true);

  // Count number of item to send to each rank
  std::vector<int> p_count(num_processes, 0);
  for (std::size_t i = 0; i < facet_cell_map.shape(0); ++i)
  {
    // Use first vertex of facet to partition into blocks
    const int dest_proc = dolfinx::MPI::index_owner(
        num_processes, facet_cell_map(i, 0) - global_min, global_range);
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
  for (std::size_t i = 0; i < facet_cell_map.shape(0); ++i)
  {
    const int dest_proc = dolfinx::MPI::index_owner(
        num_processes, facet_cell_map(i, 0) - global_min, global_range);
    xtl::span<std::int64_t> buffer = send_buffer.links(dest_proc);

    for (int j = 0; j < (num_vertices_per_facet + 1); ++j)
      buffer[pos[dest_proc] + j] = facet_cell_map(i, j);
    buffer[pos[dest_proc] + num_vertices_per_facet] += cell_offset;

    pos[dest_proc] += num_vertices_per_facet + 1;
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
                       int tdim)
{
  LOG(INFO) << "Build mesh dual graph";

  // Compute local part of dual graph
  auto [local_graph, facet_cell_map] = mesh::build_local_dual_graph(
      cell_vertices.array(), cell_vertices.offsets(), tdim);

  // Compute nonlocal part
  auto [graph, num_ghost_nodes] = compute_nonlocal_dual_graph(
      mpi_comm, cell_vertices.num_nodes(), facet_cell_map, local_graph);

  LOG(INFO) << "Graph edges (local:" << local_graph.offsets().back()
            << ", non-local:"
            << graph.offsets().back() - local_graph.offsets().back() << ")";

  return {std::move(graph), {num_ghost_nodes, local_graph.offsets().back()}};
}
//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int32_t>, xt::xtensor<std::int64_t, 2>>
mesh::build_local_dual_graph(const xtl::span<const std::int64_t>& cell_vertices,
                             const xtl::span<const std::int32_t>& offsets,
                             int tdim)
{
  LOG(INFO) << "Build local part of mesh dual graph";
  return compute_local_dual_graph_keyed(cell_vertices, offsets, tdim);
}
//-----------------------------------------------------------------------------
