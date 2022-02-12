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
  MPI_Iexscan(&num_local, &cell_offset, 1,
              dolfinx::MPI::mpi_type<std::int64_t>(), MPI_SUM, comm,
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
  // Create a return buffer with one entry per received facet
  // set to -1 for facets that do not get a match.
  std::vector<std::int64_t> return_buffer(num_facets_rcvd, -1);
  std::vector<int> return_offsets(recvd_buffer.offsets().size());
  std::transform(recvd_buffer.offsets().begin(), recvd_buffer.offsets().end(),
                 return_offsets.begin(),
                 [&max_num_vertices_per_facet](int off) {
                   return off / (max_num_vertices_per_facet + 1);
                 });

  // Reshape the received buffer
  {
    std::vector<std::int32_t> rc_off(num_facets_rcvd + 1, 0);
    for (std::size_t i = 0; i < rc_off.size() - 1; ++i)
      rc_off[i + 1] = rc_off[i] + (max_num_vertices_per_facet + 1);
    recvd_buffer = graph::AdjacencyList<std::int64_t>(
        std::move(recvd_buffer.array()), std::move(rc_off));
  }

  // Get permutation that takes facets into sorted order
  std::vector<int> perm(num_facets_rcvd);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&recvd_buffer](int a, int b) {
    return std::lexicographical_compare(
        recvd_buffer.links(a).begin(), std::prev(recvd_buffer.links(a).end()),
        recvd_buffer.links(b).begin(), std::prev(recvd_buffer.links(b).end()));
  });

  // Find matching facets
  bool this_equal, last_equal = false;
  for (int i = 1; i < num_facets_rcvd; ++i)
  {
    const int i0 = perm[i - 1];
    const int i1 = perm[i];
    const auto facet0 = recvd_buffer.links(i0);
    const auto facet1 = recvd_buffer.links(i1);
    const std::int64_t cell0 = facet0.back();
    const std::int64_t cell1 = facet1.back();

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
      assert(i0 < (int)return_buffer.size() and i1 < (int)return_buffer.size());
      return_buffer[i0] = cell1;
      return_buffer[i1] = cell0;
    }
    last_equal = this_equal;
  }

  const graph::AdjacencyList<std::int64_t> return_cells(
      std::move(return_buffer), std::move(return_offsets));

  graph::AdjacencyList<std::int64_t> returned_list
      = dolfinx::MPI::all_to_all(comm, return_cells);
  assert(returned_list.num_nodes() == send_buffer.num_nodes());
  assert(returned_list.array().size() * (max_num_vertices_per_facet + 1)
         == send_buffer.array().size());

  // Count number of adjacency list edges in local graph
  std::vector<int> edge_count(local_graph.num_nodes());
  for (int i = 0; i < local_graph.num_nodes(); ++i)
    edge_count[i] = local_graph.num_links(i);

  // Count new received entries
  const auto send_c = send_buffer.array();
  const auto recv_c = returned_list.array();
  int num_ghost_edges = 0;
  for (std::size_t i = 0; i < recv_c.size(); ++i)
  {
    const std::int64_t cell1 = recv_c[i];
    if (cell1 != -1)
    {
      const std::int64_t cell0 = send_c[i * (max_num_vertices_per_facet + 1)
                                        + max_num_vertices_per_facet]
                                 - cell_offset;
      ++edge_count[cell0];
      ++num_ghost_edges;
    }
  }

  // Build adjacency list
  offsets.assign(edge_count.size() + 1, 0);
  std::partial_sum(edge_count.begin(), edge_count.end(),
                   std::next(offsets.begin()));
  graph::AdjacencyList<std::int64_t> graph(
      std::vector<std::int64_t>(offsets.back()), std::move(offsets));
  // Copy existing local graph, adding cell_offset
  pos.assign(graph.num_nodes(), 0);
  for (int i = 0; i < local_graph.num_nodes(); ++i)
  {
    std::transform(local_graph.links(i).begin(), local_graph.links(i).end(),
                   graph.links(i).begin(),
                   [&cell_offset](std::int64_t j) { return cell_offset + j; });
    pos[i] = local_graph.num_links(i);
  }

  // Insert new entries
  for (std::size_t i = 0; i < recv_c.size(); ++i)
  {
    const std::int64_t cell1 = recv_c[i];
    if (cell1 != -1)
    {
      const std::int64_t cell0 = send_c[i * (max_num_vertices_per_facet + 1)
                                        + max_num_vertices_per_facet]
                                 - cell_offset;
      auto links0 = graph.links(cell0);
      links0[pos[cell0]++] = cell1;
    }
  }

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
                     [&cell_vertices_local, offset = cell_offsets[c]](auto fv) {
                       return cell_vertices_local[offset + fv];
                     });

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
graph::AdjacencyList<std::int64_t>
mesh::vertex_ownership(MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       const graph::AdjacencyList<std::int64_t>& dual_graph,
                       const graph::AdjacencyList<int>& cell_destinations)
{
  // Find any entries in dual-graph which are off-process
  const std::int64_t num_local_cells = cells.num_nodes();
  std::int64_t global_cell_offset = 0;
  MPI_Exscan(&num_local_cells, &global_cell_offset, 1, MPI_INT64_T, MPI_SUM,
             comm);

  std::cout << "nl = " << num_local_cells << "off = " << global_cell_offset
            << "\n";

  // Global-to-local map for vertices
  std::map<std::int64_t, int> global_to_local;
  int c = 0;

  // Start by inserting vertices which are in cells connected to off-process
  // cells. These vertices may be shared by multiple as yet unknown processes.
  // Note: this is overkill, because some vertices in boundary cells will
  // not be on the boundary.
  for (int i = 0; i < cells.num_nodes(); ++i)
  {
    for (std::int64_t q : dual_graph.links(i))
    {
      if (q < global_cell_offset or q >= global_cell_offset + num_local_cells)
      {
        for (std::int64_t v : cells.links(i))
        {
          if (global_to_local.insert({v, c}).second)
            ++c;
        }
      }
    }
  }
  const int nv_boundary = c;
  std::cout << "nv_boundary = " << nv_boundary << " \n";

  for (std::int64_t v : cells.array())
  {
    if (global_to_local.insert({v, c}).second)
      ++c;
  }
  // Reverse map, local to global
  std::vector<std::int64_t> global_index(global_to_local.size());
  for (auto q : global_to_local)
    global_index[q.second] = q.first;

  // Count number of occurrences of each vertex
  std::vector<int> count(global_index.size(), 0);
  for (int v = 0; v < nv_boundary; ++v)
    count[v] = 1;
  for (std::int64_t vglobal : cells.array())
    count[global_to_local[vglobal]]++;
  std::vector<int> offset(count.size() + 1, 0);
  std::partial_sum(count.begin(), count.end(), std::next(offset.begin()));
  std::vector<std::int32_t> owners(offset.back());

  // Add -1 as owner for boundary vertices
  for (int v = 0; v < nv_boundary; ++v)
    owners[offset[v]++] = -1;

  // Get set of known cell owners for each vertex
  for (int i = 0; i < cells.num_nodes(); ++i)
  {
    // Get owner of this cell
    const int owner = cell_destinations.links(i)[0];
    for (std::int64_t vglobal : cells.links(i))
    {
      const int v = global_to_local[vglobal];
      owners[offset[v]++] = owner;
    }
  }
  // Reset offsets
  offset[0] = 0;
  std::partial_sum(count.begin(), count.end(), std::next(offset.begin()));

  // Get unique set of owners for each vertex
  for (std::size_t j = 0; j < offset.size() - 1; ++j)
  {
    auto it_start = std::next(owners.begin(), offset[j]);
    auto it_end = std::next(owners.begin(), offset[j + 1]);
    std::sort(it_start, it_end);
    count[j] = std::distance(it_start, std::unique(it_start, it_end));
  }
  // Compress
  int new_offset_j = 0;
  for (std::size_t j = 0; j < offset.size() - 1; ++j)
  {
    std::copy(std::next(owners.begin(), offset[j]),
              std::next(owners.begin(), offset[j] + count[j]),
              std::next(owners.begin(), new_offset_j));
    offset[j] = new_offset_j;
    new_offset_j += count[j];
  }
  offset.back() = new_offset_j;
  owners.resize(offset.back());

  // We now have a map from local vertices to cell 'owners'. If in the interior
  // of our domain, we can decide the ownership of the vertex and send to
  // sharing processes. If it is just one value, there is no need to send, as
  // the owning process will know (because it owns all adjacent cells).
  // If there is a -1, then a further decision must be made after distribution.
  const graph::AdjacencyList<int> vprocs(std::move(owners), std::move(offset));

  // Iterate over vprocs
  //
  // if num_links(i) == 1 - ignore
  // otherwise decide an owner (random...) and send to all sharing processes
  //
  // What about ghost cells?
  //

  const int mpi_size = MPI::size(comm);
  std::vector<std::vector<std::int64_t>> vertex_ownership_send(mpi_size);

  for (int i = 0; i < vprocs.num_nodes(); ++i)
  {
    const std::int64_t gi = global_index[i];
    auto p = vprocs.links(i);
    if (p.size() > 1 and p[0] == -1)
    {
      for (std::size_t j = 1; j < p.size(); ++j)
      {
        int q = p[j];
        vertex_ownership_send[q].push_back(gi);
        vertex_ownership_send[q].push_back(p.size());
        for (int r : p)
          vertex_ownership_send[q].push_back(r);
      }
    }
    else if (p.size() > 1)
    {
      for (int q : p)
      {
        vertex_ownership_send[q].push_back(gi);
        vertex_ownership_send[q].push_back(p.size());
        for (int r : p)
          vertex_ownership_send[q].push_back(r);
      }
    }
  }

  // Send to owners
  std::vector<std::int64_t> vertex_owner_recv
      = MPI::all_to_all(
            comm, graph::AdjacencyList<std::int64_t>(vertex_ownership_send))
            .array();

  std::size_t p = 0;
  std::vector<int> voffset = {0};
  std::vector<std::int64_t> varr;
  while (p < vertex_owner_recv.size())
  {
    varr.push_back(vertex_owner_recv[p++]);
    const int np = vertex_owner_recv[p++];
    varr.insert(varr.end(), std::next(vertex_owner_recv.begin(), p),
                std::next(vertex_owner_recv.begin(), p + np));
    voffset.push_back(voffset.back() + np + 1);
    p += np;
  }
  return graph::AdjacencyList<std::int64_t>(std::move(varr),
                                            std::move(voffset));
}

graph::AdjacencyList<std::int64_t> mesh::vertex_ownership_part2(
    const graph::AdjacencyList<std::int64_t>& vertex_ownership,
    const graph::AdjacencyList<std::int32_t>&,
    const xt::xtensor<std::int64_t, 2>& unmatched_facets)
{
  std::set<std::int64_t> edge_verts(unmatched_facets.begin(),
                                    unmatched_facets.end());

  std::stringstream s;

  for (int i = 0; i < vertex_ownership.num_nodes(); ++i)
  {
    auto vi = vertex_ownership.links(i);
    std::int64_t v = vi[0];
    if (vi[1] == -1)
    {
      if (edge_verts.find(v) == edge_verts.end())
      {
        // Not on boundary here, so ownership should be clear
        s << "Resolved: ";
        for (auto q : vi)
          s << q << " ";
        s << "\n";
      }
      else
      {
        // Problem node...
        s << "Problem: ";
        for (auto q : vi)
          s << q << " ";
        s << "\n";
      }
    }
    else
    {
      s << "Local: ";
      for (auto q : vi)
        s << q << " ";
      s << "\n";
    }
  }

  std::cout << s.str();

  graph::AdjacencyList<std::int64_t> new_vertex_ownership = vertex_ownership;

  return new_vertex_ownership;
}