// Copyright (C) 2010-2022 Garth N. Wells
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
#include <span>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{
/// Return cell type, deduced from the number of cell vertices and the
/// topological dimension of the cell
constexpr mesh::CellType get_cell_type(int num_vertices, int tdim)
{
  switch (tdim)
  {
  case 1:
    return mesh::CellType::interval;
  case 2:
    switch (num_vertices)
    {
    case 3:
      return mesh::CellType::triangle;
    case 4:
      return mesh::CellType::quadrilateral;
    default:
      throw std::runtime_error("Invalid data");
    }
  case 3:
    switch (num_vertices)
    {
    case 4:
      return mesh::CellType::tetrahedron;
    case 5:
      return mesh::CellType::pyramid;
    case 6:
      return mesh::CellType::prism;
    case 8:
      return mesh::CellType::hexahedron;
    default:
      throw std::runtime_error("Invalid data");
    }
  default:
    throw std::runtime_error("Invalid data");
  }
}

//-----------------------------------------------------------------------------

/// @brief Build nonlocal part of dual graph for mesh and return number
/// of non-local edges.
///
/// @note Scalable version
///
/// @note graphbuild::compute_local_dual_graph should be called
/// before this function is called.
///
/// @param[in] comm MPI communicator
/// @param[in] facets Facets on this rank that are shared by only on
/// cell on this rank, i.e. candidates for possibly residing on other
/// processes. Each row in `facets` corresponds to a facet, and the row
/// data has the form [v0, ..., v_{n-1}, x, x], where `v_i` are the
/// sorted vertex global indices of the facets and `x` is a padding
/// value for the mixed topology case where facets can have differing
/// number of vertices.
/// @param[in] shape1 Number of columns for `facets`.
/// @param[in] cells Attached cell (local index) for each facet in
/// `facet`.
/// @param[in] local_graph The dual graph for cells on this MPI rank
/// @return (0) Extended dual graph to include ghost edges (edges to
/// off-procss cells) and (1) the number of ghost edges
graph::AdjacencyList<std::int64_t> compute_nonlocal_dual_graph(
    const MPI_Comm comm, std::span<const std::int64_t> facets,
    std::size_t shape1, std::span<const std::int32_t> cells,
    const graph::AdjacencyList<std::int32_t>& local_graph)
{
  LOG(INFO) << "Build nonlocal part of mesh dual graph";
  common::Timer timer("Compute non-local part of mesh dual graph");

  // TODO: Two possible straightforward optimisations:
  // 1. Do not send owned data to self via MPI.
  // 2. Modify MPI::index_owner to use a subset of ranks as post offices.
  // 3. Find the max buffer row size for the neighbourhood rather than
  //    globally.
  //
  // Less straightforward optimisations:
  // 4. After matching, send back matches only, (and only to ranks with
  //    a match) (Note: this would complicate the communication and
  //    handling of buffers)

  const std::size_t shape0 = cells.size();

  // Return empty data if mesh is not distributed
  const int num_ranks = dolfinx::MPI::size(comm);
  if (num_ranks == 1)
  {
    // Convert graph to int64_t and return
    return graph::AdjacencyList<std::int64_t>(
        std::vector<std::int64_t>(local_graph.array().begin(),
                                  local_graph.array().end()),
        local_graph.offsets());
  }

  // Get cell offset for this process for converting local cell indices
  // to global cell indices
  std::int64_t cell_offset = 0;
  MPI_Request request_cell_offset;
  {
    const std::int64_t num_local = local_graph.num_nodes();
    MPI_Iexscan(&num_local, &cell_offset, 1, MPI_INT64_T, MPI_SUM, comm,
                &request_cell_offset);
  }

  // Find (max_vert_per_facet, min_vertex_index, max_vertex_index)
  // across all processes. Use first facet vertex for min/max index.
  std::int32_t fshape1 = -1;
  std::array<std::int64_t, 2> vrange;
  {
    std::array<std::int64_t, 3> send_buffer_r
        = {std::int64_t(shape1), std::numeric_limits<std::int64_t>::min(), -1};
    for (std::size_t i = 0; i < facets.size(); i += shape1)
    {
      send_buffer_r[1] = std::max(send_buffer_r[1], -facets[i]);
      send_buffer_r[2] = std::max(send_buffer_r[2], facets[i]);
    }

    // Compute reductions
    std::array<std::int64_t, 3> recv_buffer_r;
    MPI_Allreduce(send_buffer_r.data(), recv_buffer_r.data(), 3, MPI_INT64_T,
                  MPI_MAX, comm);
    assert(recv_buffer_r[1] != std::numeric_limits<std::int64_t>::min());
    assert(recv_buffer_r[2] != -1);
    fshape1 = recv_buffer_r[0];
    vrange = {-recv_buffer_r[1], recv_buffer_r[2] + 1};

    LOG(2) << "Max. vertices per facet=" << fshape1 << "\n";
  }
  const std::int32_t buffer_shape1 = fshape1 + 1;

  // Build list of dest ranks and count number of items (facets) to send
  // to each dest post office (by neighbourhood rank)
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest, pos_to_neigh_rank(shape0, -1);
  {
    // Build {dest, pos} list for each facet, and sort (dest is the post
    // office rank)
    std::vector<std::array<std::int32_t, 2>> dest_to_index;
    dest_to_index.reserve(shape0);
    std::int64_t range = vrange[1] - vrange[0];
    for (std::size_t i = 0; i < shape0; ++i)
    {
      std::int64_t v0 = facets[i * shape1] - vrange[0];
      dest_to_index.push_back({dolfinx::MPI::index_owner(num_ranks, v0, range),
                               static_cast<int>(i)});
    }
    std::sort(dest_to_index.begin(), dest_to_index.end());

    // Build list of dest ranks and count number of items (facets) to
    // send to each dest post office (by neighbourhood rank)
    {
      auto it = dest_to_index.begin();
      while (it != dest_to_index.end())
      {
        const int neigh_rank = dest.size();

        // Store global rank
        dest.push_back((*it)[0]);

        // Find iterator to next global rank
        auto it1 = std::find_if(it, dest_to_index.end(),
                                [r = dest.back()](auto& idx)
                                { return idx[0] != r; });

        // Store number of items for current rank
        num_items_per_dest.push_back(std::distance(it, it1));

        // Set entry in map from local facet row index (position) to local
        // destination rank
        for (auto e = it; e != it1; ++e)
          pos_to_neigh_rank[(*e)[1]] = neigh_rank;

        // Advance iterator
        it = it1;
      }
    }
  }

  // Determine source ranks
  const std::vector<int> src
      = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  LOG(INFO) << "Number of destination and source ranks in non-local dual graph "
               "construction, and ratio to total number of ranks: "
            << dest.size() << ", " << src.size() << ", "
            << static_cast<double>(dest.size()) / num_ranks << ", "
            << static_cast<double>(src.size()) / num_ranks;

  // Create neighbourhood communicator for sending data to
  // post offices
  MPI_Comm neigh_comm0;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm0);

  // Compute send displacements
  std::vector<std::int32_t> send_disp(num_items_per_dest.size() + 1, 0);
  std::partial_sum(num_items_per_dest.begin(), num_items_per_dest.end(),
                   std::next(send_disp.begin()));

  // Wait for the MPI_Iexscan to complete (before using cell_offset)
  MPI_Wait(&request_cell_offset, MPI_STATUS_IGNORE);

  // Pack send buffer
  std::vector<std::int32_t> send_indx_to_pos(send_disp.back());
  std::vector<std::int64_t> send_buffer(buffer_shape1 * send_disp.back(), -1);
  {
    std::vector<std::int32_t> send_offsets = send_disp;
    for (std::size_t i = 0; i < shape0; ++i)
    {
      int neigh_dest = pos_to_neigh_rank[i];
      std::size_t pos = send_offsets[neigh_dest];
      send_indx_to_pos[pos] = i;

      // Copy facet data into buffer
      std::copy_n(std::next(facets.begin(), i * shape1), shape1,
                  std::next(send_buffer.begin(), buffer_shape1 * pos));
      send_buffer[buffer_shape1 * pos + fshape1] = cells[i] + cell_offset;
      ++send_offsets[neigh_dest];
    }
  }

  // Send number of send items to post offices
  std::vector<int> num_items_recv(src.size());
  num_items_per_dest.reserve(1);
  num_items_recv.reserve(1);
  MPI_Neighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                        num_items_recv.data(), 1, MPI_INT, neigh_comm0);

  // Prepare receive displacement and buffers
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
                         neigh_comm0);

  MPI_Type_free(&compound_type);
  MPI_Comm_free(&neigh_comm0);

  // Search for consecutive facets (-> dual graph edge between cells)
  // and pack into send buffer
  std::vector<std::int64_t> send_buffer1(recv_disp.back(), -1);
  {
    // Compute sort permutation for received data
    std::vector<int> sort_order(recv_buffer.size() / buffer_shape1);
    std::iota(sort_order.begin(), sort_order.end(), 0);
    std::sort(sort_order.begin(), sort_order.end(),
              [&recv_buffer, buffer_shape1, fshape1](auto f0, auto f1)
              {
                auto it0 = std::next(recv_buffer.begin(), f0 * buffer_shape1);
                auto it1 = std::next(recv_buffer.begin(), f1 * buffer_shape1);
                return std::lexicographical_compare(
                    it0, std::next(it0, fshape1), it1, std::next(it1, fshape1));
              });

    auto it = sort_order.begin();
    while (it != sort_order.end())
    {
      std::size_t offset0 = (*it) * buffer_shape1;
      auto f0 = std::next(recv_buffer.data(), offset0);

      // Find iterator to next facet different from f0
      auto it1 = std::find_if_not(
          it, sort_order.end(),
          [f0, &recv_buffer, buffer_shape1, fshape1](auto idx) -> bool
          {
            std::size_t offset1 = idx * buffer_shape1;
            auto f1 = std::next(recv_buffer.data(), offset1);
            return std::equal(f0, std::next(f0, fshape1), f1);
          });

      std::size_t num_matches = std::distance(it, it1);
      if (num_matches > 2)
      {
        throw std::runtime_error(
            "A facet is connected to more than two cells.");
      }

      // TODO: generalise for more than matches and log warning (maybe
      // with an option?). Would need to send back multiple values.
      if (num_matches == 2)
      {
        // Store the global cell index from the other rank
        send_buffer1[*it] = recv_buffer[*(it + 1) * buffer_shape1 + fshape1];
        send_buffer1[*(it + 1)] = recv_buffer[*it * buffer_shape1 + fshape1];
      }

      // Advance iterator and increment entity
      it = it1;
    }
  }

  // Create neighbourhood communicator for sending data from post
  // offices
  MPI_Comm neigh_comm1;
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm1);

  // Send back data
  std::vector<std::int64_t> recv_buffer1(send_disp.back());
  MPI_Neighbor_alltoallv(send_buffer1.data(), num_items_recv.data(),
                         recv_disp.data(), MPI_INT64_T, recv_buffer1.data(),
                         num_items_per_dest.data(), send_disp.data(),
                         MPI_INT64_T, neigh_comm1);
  MPI_Comm_free(&neigh_comm1);

  // --- Build new graph

  // Count number of adjacency list edges
  std::vector<std::int32_t> num_edges(local_graph.num_nodes(), 0);
  std::adjacent_difference(std::next(local_graph.offsets().begin()),
                           local_graph.offsets().end(), num_edges.begin());
  for (std::size_t i = 0; i < recv_buffer1.size(); ++i)
  {
    if (recv_buffer1[i] >= 0)
    {
      std::size_t pos = send_indx_to_pos[i];
      std::size_t cell = cells[pos];
      num_edges[cell] += 1;
    }
  }

  // Compute adjacency list offsets
  std::vector<std::int32_t> offsets(local_graph.num_nodes() + 1, 0);
  std::partial_sum(num_edges.cbegin(), num_edges.cend(),
                   std::next(offsets.begin()));

  // Compute adjacency list data (edges)
  std::vector<std::int64_t> data(offsets.back());
  {
    std::vector<std::int32_t> disp = offsets;

    // Copy local data and add cell offset
    for (std::int32_t i = 0; i < local_graph.num_nodes(); ++i)
    {
      auto e = local_graph.links(i);
      disp[i] += e.size();
      std::transform(e.begin(), e.end(), std::next(data.begin(), offsets[i]),
                     [cell_offset](auto x) { return x + cell_offset; });
    }

    // Add non-local data
    for (std::size_t i = 0; i < recv_buffer1.size(); ++i)
    {
      if (recv_buffer1[i] >= 0)
      {
        std::size_t pos = send_indx_to_pos[i];
        std::size_t cell = cells[pos];
        data[disp[cell]++] = recv_buffer1[i];
      }
    }
  }

  return graph::AdjacencyList<std::int64_t>(std::move(data),
                                            std::move(offsets));
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>,
           std::size_t, std::vector<std::int32_t>>
mesh::build_local_dual_graph(std::span<const std::int64_t> cell_vertices,
                             std::span<const std::int32_t> cell_offsets,
                             int tdim)
{
  LOG(INFO) << "Build local part of mesh dual graph";
  common::Timer timer("Compute local part of mesh dual graph");

  const std::int32_t num_cells = cell_offsets.size() - 1;
  if (num_cells == 0)
  {
    // Empty mesh on this process
    return {graph::AdjacencyList<std::int32_t>(0), std::vector<std::int64_t>(),
            0, std::vector<std::int32_t>()};
  }

  // Note: only meshes with a single cell type are supported
  const graph::AdjacencyList<int> cell_facets = mesh::get_entity_vertices(
      get_cell_type(cell_offsets[1] - cell_offsets[0], tdim), tdim - 1);

  // Determine maximum number of vertices for facet
  int max_vertices_per_facet = 0;
  for (int i = 0; i < cell_facets.num_nodes(); ++i)
  {
    max_vertices_per_facet
        = std::max(max_vertices_per_facet, cell_facets.num_links(i));
  }

  const int shape1 = max_vertices_per_facet + 1;

  // Build a list of facets, defined by sorted vertices, with the connected
  // cell index after the vertices
  std::vector<std::int64_t> facets;
  facets.reserve(num_cells * cell_facets.num_nodes() * shape1);
  for (auto it = cell_offsets.begin(); it != std::prev(cell_offsets.end());
       ++it)
  {
    int num_vertices = *std::next(it) - *it;
    auto v = cell_vertices.subspan(*it, num_vertices);
    std::size_t c = std::distance(cell_offsets.begin(), it);

    // Loop over cell facets
    for (int f = 0; f < cell_facets.num_nodes(); ++f)
    {
      auto facet_vertices = cell_facets.links(f);
      std::transform(facet_vertices.begin(), facet_vertices.end(),
                     std::back_inserter(facets),
                     [v](auto idx) { return v[idx]; });
      std::sort(std::prev(facets.end(), facet_vertices.size()), facets.end());
      facets.insert(facets.end(),
                    max_vertices_per_facet - facet_vertices.size(), -1);
      facets.push_back(c);
    }
  }

  // Sort facets by vertex key
  std::vector<std::size_t> perm(facets.size() / shape1, 0);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&facets, shape1](auto f0, auto f1)
            {
              auto it0 = std::next(facets.begin(), f0 * shape1);
              auto it1 = std::next(facets.begin(), f1 * shape1);
              return std::lexicographical_compare(it0, std::next(it0, shape1),
                                                  it1, std::next(it1, shape1));
            });

  // Iterate over sorted list of facets. Facets shared by more than one
  // cell lead to a graph edge to be added. Facets that are not shared
  // are stored as these might be shared by a cell on another process.
  std::vector<std::int64_t> unmatched_facets;
  std::vector<std::int32_t> cells;
  std::vector<std::array<std::int32_t, 2>> edges;
  {
    auto it = perm.begin();
    while (it != perm.end())
    {
      auto f0 = std::span(facets.data() + (*it) * shape1, shape1);

      // Find iterator to next facet different from f0
      auto it1 = std::find_if_not(
          it, perm.end(),
          [f0, &facets, shape1](auto idx) -> bool
          {
            auto f1_it = std::next(facets.begin(), idx * shape1);
            return std::equal(f0.begin(), std::prev(f0.end()), f1_it);
          });

      // Add dual graph edges (one direction only, other direction is
      // added later)
      std::int32_t cell0 = f0.back();
      for (auto itx = std::next(it); itx != it1; ++itx)
      {
        auto f1 = std::span(facets.data() + *itx * shape1, shape1);
        std::int32_t cell1 = f1.back();
        edges.push_back({cell0, cell1});
      }

      // Store unmatched facets and the attached cell
      if (std::distance(it, it1) == 1)
      {
        unmatched_facets.insert(unmatched_facets.end(), f0.begin(),
                                std::prev(f0.end()));
        cells.push_back(cell0);
      }

      // Update iterator
      it = it1;
    }
  }

  // -- Build adjacency list data

  std::vector<std::int32_t> sizes(num_cells, 0);
  for (auto e : edges)
  {
    ++sizes[e[0]];
    ++sizes[e[1]];
  }

  std::vector<std::int32_t> offsets(sizes.size() + 1, 0);
  std::partial_sum(sizes.cbegin(), sizes.cend(), std::next(offsets.begin()));
  std::vector<std::int32_t> data(offsets.back());
  {
    std::vector<std::int32_t> pos = offsets;
    for (auto e : edges)
    {
      data[pos[e[0]]++] = e[1];
      data[pos[e[1]]++] = e[0];
    }
  }

  return {
      graph::AdjacencyList<std::int32_t>(std::move(data), std::move(offsets)),
      std::move(unmatched_facets), max_vertices_per_facet, std::move(cells)};
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int64_t>
mesh::build_dual_graph(const MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       int tdim)
{
  LOG(INFO) << "Building mesh dual graph";

  // Compute local part of dual graph (cells are graph nodes, and edges
  // are connections by facet)
  auto [local_graph, facets, shape1, fcells]
      = mesh::build_local_dual_graph(cells.array(), cells.offsets(), tdim);
  assert(local_graph.num_nodes() == cells.num_nodes());

  // Extend with nonlocal edges and convert to global indices
  graph::AdjacencyList<std::int64_t> graph
      = compute_nonlocal_dual_graph(comm, facets, shape1, fcells, local_graph);

  LOG(INFO) << "Graph edges (local: " << local_graph.offsets().back()
            << ", non-local: "
            << graph.offsets().back() - local_graph.offsets().back() << ")";

  return graph;
}
//-----------------------------------------------------------------------------
