// Copyright (C) 2010-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "graphbuild.h"
#include "cell_types.h"
#include <algorithm>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <utility>
#include <vector>
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------

/// Build nonlocal part of dual graph for mesh and return number of
/// non-local edges. Note: GraphBuilder::compute_local_dual_graph should
/// be called before this function is called. Returns (ghost vertices,
/// num_nonlocal_edges)
/// @param[in] comm MPI communicator
/// @param[in] unmatched_facets Facets on this rank that are shared by
/// only on cell on this rank. This makes them candidates for possibly
/// matching to the same facet on another MPI rank. Each row
/// `unmatched_facets` corresponds to a facet, and the row data has the
/// form [v0, ..., v_{n-1}, x, x, cell_index], where `v_i` are the
/// sorted vertex global indices of the facets, `x` is a padding value
/// for the mixed topology case where facets can have differing number
/// of vertices, and `cell_index` is the global index of the attached
/// cell.
/// @param[in] local_graph The dual graph for cells on this MPI rank
/// @return (0) Extended dual graph to include ghost edges (edges to
/// off-rank cells) and (1) the number of ghost edges
std::pair<graph::AdjacencyList<std::int64_t>, std::int32_t>
compute_nonlocal_dual_graph(
    const MPI_Comm comm, const xt::xtensor<std::int64_t, 2>& unmatched_facets,
    const graph::AdjacencyList<std::int32_t>& local_graph)
{
  LOG(INFO) << "Build nonlocal part of mesh dual graph";
  common::Timer timer("Compute non-local part of mesh dual graph");

  // Get number of MPI processes, and return if mesh is not distributed
  const int num_ranks = dolfinx::MPI::size(comm);
  if (num_ranks == 1)
  {
    // Convert graph to int64
    return {graph::AdjacencyList<std::int64_t>(
                std::vector<std::int64_t>(local_graph.array().begin(),
                                          local_graph.array().end()),
                local_graph.offsets()),
            0};
  }

  // Get cell offset for this process to create global numbering for
  // cells
  const std::int64_t num_local = local_graph.num_nodes();
  std::int64_t cell_offset = 0;
  MPI_Request request_cell_offset;
  MPI_Iexscan(&num_local, &cell_offset, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_cell_offset);

  // At this stage facet_cell map only contains facets->cells with edge
  // facets either interprocess or external boundaries

  // TODO: improve scalability, possibly by limiting the number of
  // processes which do the matching, and using a neighbor comm?

  // (0) Some ranks may have empty unmatched_facets, so get max across
  // all ranks
  // (1) Find the global range of the first vertex index of each facet
  // in the list and use this to divide up the facets between all
  // processes.
  //
  // Combine into single MPI reduce (MPI_MIN)
  std::array<std::int64_t, 3> buffer_local_min
      = {-std::int64_t(unmatched_facets.shape(1) - 1),
         std::numeric_limits<std::int64_t>::max(), 0};
  if (unmatched_facets.shape(0) > 0)
  {
    auto local_minmax = xt::minmax(xt::col(unmatched_facets, 0))();
    buffer_local_min[1] = local_minmax[0];
    buffer_local_min[2] = -local_minmax[1];
  }
  std::array<std::int64_t, 3> buffer_global_min;
  MPI_Allreduce(buffer_local_min.data(), buffer_global_min.data(), 3,
                MPI_INT64_T, MPI_MIN, comm);
  const std::int32_t max_num_vertices_per_facet = -buffer_global_min[0];
  LOG(INFO) << "Max. vertices per facet=" << max_num_vertices_per_facet << "\n";
  assert(buffer_global_min[1] != std::numeric_limits<std::int64_t>::max());
  const std::array<std::int64_t, 2> global_minmax
      = {buffer_global_min[1], -buffer_global_min[2]};
  const std::int64_t global_range = global_minmax[1] - global_minmax[0] + 1;

  // Send facet-to-cell data to intermediary match-making ranks

  // Count number of item to send to each rank
  std::vector<int> p_count(num_ranks, 0);
  for (std::size_t i = 0; i < unmatched_facets.shape(0); ++i)
  {
    // Use first vertex of facet to partition into blocks
    std::int64_t v0 = unmatched_facets(i, 0) - global_minmax[0];
    const int dest = dolfinx::MPI::index_owner(num_ranks, v0, global_range);
    p_count[dest] += max_num_vertices_per_facet + 1;
  }

  // Create back adjacency list send buffer
  std::vector<std::int32_t> offsets(num_ranks + 1, 0);
  std::partial_sum(p_count.begin(), p_count.end(), std::next(offsets.begin()));
  graph::AdjacencyList<std::int64_t> send_buffer(
      std::vector<std::int64_t>(offsets.back()), std::move(offsets));

  // Wait for the MPI_Iexscan to complete
  MPI_Wait(&request_cell_offset, MPI_STATUS_IGNORE);

  // Pack facet-to-cell to send to match-maker rank
  std::vector<int> pos(send_buffer.num_nodes(), 0);
  for (std::size_t i = 0; i < unmatched_facets.shape(0); ++i)
  {
    std::int64_t v0 = unmatched_facets(i, 0) - global_minmax[0];
    const int dest = dolfinx::MPI::index_owner(num_ranks, v0, global_range);

    // Pack facet vertices, and attached cell local index
    xtl::span<std::int64_t> buffer = send_buffer.links(dest);
    for (int j = 0; j < max_num_vertices_per_facet + 1; ++j)
      buffer[pos[dest] + j] = unmatched_facets(i, j);

    // Add cell index offset
    buffer[pos[dest] + max_num_vertices_per_facet] += cell_offset;
    pos[dest] += max_num_vertices_per_facet + 1;
  }

  // Send data
  graph::AdjacencyList<std::int64_t> recvd_buffer
      = dolfinx::MPI::all_to_all(comm, send_buffer);
  assert(recvd_buffer.array().size() % (max_num_vertices_per_facet + 1) == 0);

  // Number of received facets
  const int num_facets_rcvd
      = recvd_buffer.array().size() / (max_num_vertices_per_facet + 1);

  // Build array from received facet to source rank
  const std::vector<std::int32_t>& recvd_disp = recvd_buffer.offsets();
  std::vector<int> proc(num_facets_rcvd);
  for (int p = 0; p < num_ranks; ++p)
  {
    for (int f = recvd_disp[p] / (max_num_vertices_per_facet + 1);
         f < recvd_disp[p + 1] / (max_num_vertices_per_facet + 1); ++f)
    {
      proc[f] = p;
    }
  }

  // Reshape the received buffer
  {
    std::vector<std::int32_t> offsets(num_facets_rcvd + 1, 0);
    for (std::size_t i = 0; i < offsets.size() - 1; ++i)
      offsets[i + 1] = offsets[i] + (max_num_vertices_per_facet + 1);
    recvd_buffer = graph::AdjacencyList<std::int64_t>(
        std::move(recvd_buffer.array()), std::move(offsets));
  }

  // Get permutation that takes facets into sorted order
  std::vector<int> perm(num_facets_rcvd);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&recvd_buffer](int a, int b)
            {
              return std::lexicographical_compare(
                  recvd_buffer.links(a).begin(),
                  std::prev(recvd_buffer.links(a).end()),
                  recvd_buffer.links(b).begin(),
                  std::prev(recvd_buffer.links(b).end()));
            });

  // Count data items to send to each rank
  p_count.assign(num_ranks, 0);
  bool this_equal, last_equal = false;
  std::vector<std::int8_t> facet_match(num_facets_rcvd, false);
  for (int i = 1; i < num_facets_rcvd; ++i)
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
  offsets.assign(num_ranks + 1, 0);
  std::partial_sum(p_count.begin(), p_count.end(), std::next(offsets.begin()));
  send_buffer = graph::AdjacencyList<std::int64_t>(
      std::vector<std::int64_t>(offsets.back()), std::move(offsets));

  pos.assign(send_buffer.num_nodes(), 0);
  for (int i = 1; i < num_facets_rcvd; ++i)
  {
    if (facet_match[i])
    {
      const int i0 = perm[i - 1];
      const int i1 = perm[i];
      const int proc0 = proc[i0];
      const int proc1 = proc[i1];
      const auto facet0 = recvd_buffer.links(i0);
      const auto facet1 = recvd_buffer.links(i1);

      const std::int64_t cell0 = facet0.back();
      const std::int64_t cell1 = facet1.back();

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
    assert(cell_list[i] - cell_offset < std::int64_t(edge_count.size()));
    edge_count[cell_list[i] - cell_offset] += 1;
  }

  // Build adjacency list
  offsets.assign(edge_count.size() + 1, 0);
  std::partial_sum(edge_count.begin(), edge_count.end(),
                   std::next(offsets.begin()));
  graph::AdjacencyList<std::int64_t> graph(
      std::vector<std::int64_t>(offsets.back()), std::move(offsets));
  pos.assign(graph.num_nodes(), 0);
  std::vector<std::int64_t> ghost_edges;
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
#ifndef NDEBUG
    if (auto it_end = std::next(edges.begin(), pos[node]);
        std::find(edges.begin(), it_end, cell_list[i + 1]) != it_end)
    {
      LOG(ERROR) << "Received same edge twice in dual graph";
      throw std::runtime_error("Inconsistent mesh data in GraphBuilder: "
                               "received same edge twice in dual graph");
    }
#endif
    edges[pos[node]++] = cell_list[i + 1];
    ghost_edges.push_back(cell_list[i + 1]);
  }

  std::sort(ghost_edges.begin(), ghost_edges.end());
  const std::int32_t num_ghost_edges = std::distance(
      ghost_edges.begin(), std::unique(ghost_edges.begin(), ghost_edges.end()));

  return {std::move(graph), num_ghost_edges};
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int32_t>, xt::xtensor<std::int64_t, 2>>
mesh::build_local_dual_graph(const xtl::span<const std::int64_t>& cell_vertices,
                             const xtl::span<const std::int32_t>& cell_offsets,
                             int tdim)
{
  LOG(INFO) << "Build local part of mesh dual graph";
  common::Timer timer("Compute local part of mesh dual graph");

  const std::int32_t num_local_cells = cell_offsets.size() - 1;
  if (num_local_cells == 0)
  {
    // Empty mesh on this process
    return {graph::AdjacencyList<std::int32_t>(0),
            xt::xtensor<std::int64_t, 2>({0, 0})};
  }

  // Create local (starting from 0), contiguous version of cell_vertices
  // such that cell_vertices_local[i] and cell_vertices[i] refer to the
  // same 'vertex', but cell_vertices_local[i] uses a contiguous vertex
  // numbering that starts from 0. Note that the local vertex indices
  // are ordered, i.e. if cell_vertices[i] < cell_vertices[j] , then
  // cell_vertices_local[i] < cell_vertices_local[j].
  std::vector<std::int32_t> perm(cell_vertices.size());
  std::iota(perm.begin(), perm.end(), 0);
  dolfinx::argsort_radix<std::int64_t, 16>(cell_vertices, perm);

  std::vector<std::int32_t> cell_vertices_local(cell_vertices.size(), 0);
  std::int32_t vcounter = 0;
  for (std::size_t i = 1; i < cell_vertices.size(); ++i)
  {
    if (cell_vertices[perm[i - 1]] != cell_vertices[perm[i]])
      vcounter++;
    cell_vertices_local[perm[i]] = vcounter;
  }
  const std::int32_t num_vertices = vcounter + 1;

  // Build local-to-global map for vertices
  std::vector<std::int64_t> local_to_global_v(num_vertices);
  for (std::size_t i = 0; i < cell_vertices_local.size(); i++)
    local_to_global_v[cell_vertices_local[i]] = cell_vertices[i];

  // Count number of cells of each type, based on the number of vertices
  // in each cell, covering interval(2) through to hex(8)
  std::array<int, 9> num_cells_of_type;
  std::fill(num_cells_of_type.begin(), num_cells_of_type.end(), 0);
  for (auto it = cell_offsets.cbegin(); it != std::prev(cell_offsets.cend());
       ++it)
  {
    const std::size_t num_cell_vertices = *std::next(it) - *it;
    assert(num_cell_vertices < num_cells_of_type.size());
    ++num_cells_of_type[num_cell_vertices];
  }

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

  int num_facets = 0;
  int max_num_facet_vertices = 0;
  switch (tdim)
  {
  case 1:
    if (num_cells_of_type[2] != num_local_cells)
      throw std::runtime_error("Invalid cells in 1D mesh");
    nv_to_facets[2] = mesh::get_entity_vertices(mesh::CellType::interval, 0);
    max_num_facet_vertices = 1;
    num_facets = 2 * num_cells_of_type[2];
    break;
  case 2:
    if (num_cells_of_type[3] + num_cells_of_type[4] != num_local_cells)
      throw std::runtime_error("Invalid cells in 2D mesh");
    nv_to_facets[3] = mesh::get_entity_vertices(mesh::CellType::triangle, 1);
    nv_to_facets[4]
        = mesh::get_entity_vertices(mesh::CellType::quadrilateral, 1);
    max_num_facet_vertices = 2;
    num_facets = 3 * num_cells_of_type[3] + 4 * num_cells_of_type[4];
    break;
  case 3:
    if (num_cells_of_type[4] + num_cells_of_type[5] + num_cells_of_type[6]
            + num_cells_of_type[8]
        != num_local_cells)
    {
      throw std::runtime_error("Invalid cells in 3D mesh");
    }

    // If any quad facets in mesh, expand to width=4
    if (num_cells_of_type[5] > 0 or num_cells_of_type[6] > 0
        or num_cells_of_type[8] > 0)
    {
      max_num_facet_vertices = 4;
    }
    else
      max_num_facet_vertices = 3;

    num_facets = 4 * num_cells_of_type[4] + 5 * num_cells_of_type[5]
                 + 5 * num_cells_of_type[6] + 6 * num_cells_of_type[8];
    nv_to_facets[4] = mesh::get_entity_vertices(mesh::CellType::tetrahedron, 2);
    nv_to_facets[5] = mesh::get_entity_vertices(mesh::CellType::pyramid, 2);
    nv_to_facets[6] = mesh::get_entity_vertices(mesh::CellType::prism, 2);
    nv_to_facets[8] = mesh::get_entity_vertices(mesh::CellType::hexahedron, 2);
    break;
  default:
    throw std::runtime_error("Invalid tdim");
  }

  // Iterating over every cell, create a 'key' (sorted vertex indices)
  // for each facet and store the associated cell index
  xt::xtensor<std::int32_t, 2> facets(
      {std::size_t(num_facets), std::size_t(max_num_facet_vertices)},
      std::numeric_limits<std::int32_t>::max());
  std::vector<std::int32_t> facet_to_cell;
  facet_to_cell.reserve(num_facets);
  for (std::int32_t c = 0; c < num_local_cells; ++c)
  {
    // Cell facets (local) for current cell type
    const int num_cell_vertices = cell_offsets[c + 1] - cell_offsets[c];
    const graph::AdjacencyList<int>& cell_facets
        = nv_to_facets[num_cell_vertices];

    // Loop over all facets of cell c
    for (int f = 0; f < cell_facets.num_nodes(); ++f)
    {
      // Get data array for this facet
      auto facet = xt::row(facets, facet_to_cell.size());

      // Get facet vertices (local indices)
      auto facet_vertices = cell_facets.links(f);
      assert(facet_vertices.size() <= std::size_t(max_num_facet_vertices));
      std::transform(facet_vertices.cbegin(), facet_vertices.cend(),
                     facet.begin(),
                     [&cell_vertices_local, offset = cell_offsets[c]](auto fv)
                     { return cell_vertices_local[offset + fv]; });

      // Sort facet "indices"
      std::sort(facet.begin(), facet.end());

      // Store cell index
      facet_to_cell.push_back(c);
    }
  }
  assert(facet_to_cell.size() == facets.shape(0));

  // Sort facets by lexicographic order of vertices
  const std::vector<std::int32_t> facet_perm = dolfinx::sort_by_perm(facets);

  // Iterator over facets, and push back cells that share the facet. If
  // facet is not shared, store in 'unshared_facets'.
  std::vector<std::int32_t> edges;
  edges.reserve(num_local_cells * 2);
  std::vector<std::int32_t> unshared_facets;
  unshared_facets.reserve(num_local_cells);
  int eq_count = 0;
  for (std::size_t f = 1; f < facets.shape(0); ++f)
  {
    auto current = xt::row(facets, facet_perm[f]);
    auto previous = xt::row(facets, facet_perm[f - 1]);
    if (current == previous)
    {
      // Add cell indices
      edges.push_back(facet_to_cell[facet_perm[f]]);
      edges.push_back(facet_to_cell[facet_perm[f - 1]]);

      ++eq_count;
      if (eq_count > 1)
        LOG(WARNING) << "Same facet in more than two cells";
    }
    else
    {
      if (eq_count == 0)
        unshared_facets.push_back(facet_perm[f - 1]);
      eq_count = 0;
    }
  }

  // Add last facet if not shared
  if (eq_count == 0)
    unshared_facets.push_back(facet_perm.back());

  // Pack 'unmatched' facet data, storing facet global vertices and
  // the attached cell index
  xt::xtensor<std::int64_t, 2> unmatched_facet_data(
      {unshared_facets.size(), std::size_t(max_num_facet_vertices + 1)},
      std::numeric_limits<std::int64_t>::max());
  for (auto f = unshared_facets.begin(); f != unshared_facets.end(); ++f)
  {
    std::size_t pos = std::distance(unshared_facets.begin(), f);
    auto facet_unmatched = xt::row(unmatched_facet_data, pos);
    auto facet = xt::row(facets, *f);
    for (int v = 0; v < max_num_facet_vertices; ++v)
    {
      // Note: Global vertex indices in facet will be sorted because
      // xt::row(facets, *f) is sorted, and since if cell_vertices[i] <
      // cell_vertices[j]  then cell_vertices_local[i] <
      // cell_vertices_local[j].
      if (std::int32_t vertex = facet[v]; vertex < num_vertices)
        facet_unmatched[v] = local_to_global_v[vertex];
    }

    // Store cell index
    facet_unmatched.back() = facet_to_cell[*f];
  }

  // Count number of edges for each cell
  std::vector<std::int32_t> num_edges(num_local_cells, 0);
  for (std::int32_t cell : edges)
  {
    assert(cell < num_local_cells);
    ++num_edges[cell];
  }

  // Compute adjacency list offsets
  std::vector<std::int32_t> offsets(num_edges.size() + 1, 0);
  std::partial_sum(num_edges.begin(), num_edges.end(),
                   std::next(offsets.begin()));

  // Build adjacency data
  std::vector<std::int32_t> local_graph_data(offsets.back());
  std::vector<std::int32_t> pos(offsets.begin(), std::prev(offsets.end()));
  for (std::size_t e = 0; e < edges.size(); e += 2)
  {
    const std::size_t c0 = edges[e];
    const std::size_t c1 = edges[e + 1];
    assert(c0 < pos.size());
    assert(c1 < pos.size());
    assert(pos[c0] < (int)local_graph_data.size());
    assert(pos[c1] < (int)local_graph_data.size());
    local_graph_data[pos[c0]++] = c1;
    local_graph_data[pos[c1]++] = c0;
  }

  return {graph::AdjacencyList<std::int32_t>(std::move(local_graph_data),
                                             std::move(offsets)),
          std::move(unmatched_facet_data)};
}
//-----------------------------------------------------------------------------
std::pair<graph::AdjacencyList<std::int64_t>, std::int32_t>
mesh::build_dual_graph(const MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       int tdim)
{
  LOG(INFO) << "Build mesh dual graph";

  // Compute local part of dual graph (cells are graph nodes, and edges
  // are connections by facet)
  auto [local_graph, facet_cell_map]
      = mesh::build_local_dual_graph(cells.array(), cells.offsets(), tdim);

  // Extend with nonlocal edges and convert to global indices
  auto [graph, num_ghost_edges]
      = compute_nonlocal_dual_graph(comm, facet_cell_map, local_graph);
  assert(local_graph.num_nodes() == cells.num_nodes());

  LOG(INFO) << "Graph edges (local:" << local_graph.offsets().back()
            << ", non-local:"
            << graph.offsets().back() - local_graph.offsets().back() << ")";

  return {std::move(graph), num_ghost_edges};
}
//-----------------------------------------------------------------------------
