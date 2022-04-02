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
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

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
};

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
    const MPI_Comm comm, const xtl::span<const std::int64_t>& facets,
    std::size_t shape1, const xtl::span<const std::int32_t>& cells,
    const graph::AdjacencyList<std::int32_t>& local_graph)
{
  LOG(INFO) << "Build nonlocal part of mesh dual graph";
  common::Timer timer("Compute non-local part of mesh dual graph");

  // TODO: Two possible straightforward optimisations:
  // 1. Do not send owned data to self via MPI.
  // 2. Modify MPI::index_owner to use a subet of ranks as post offices.
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
      std::transform(e.cbegin(), e.cend(), std::next(data.begin(), offsets[i]),
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
mesh::build_local_dual_graph_new(
    const xtl::span<const std::int64_t>& cell_vertices,
    const xtl::span<const std::int32_t>& cell_offsets, int tdim)
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

  // Determine maximum number of vertices for a cell
  int max_vertices_per_cell = std::transform_reduce(
      std::next(cell_offsets.begin()), cell_offsets.end(), cell_offsets.begin(),
      0, [](auto n0, auto n1) { return std::max(n0, n1); },
      [](auto idx1, auto idx0) { return idx1 - idx0; });

  std::cout << "Test A: " << max_vertices_per_cell << std::endl;

  // Determine maximum number of vertices for facet
  const graph::AdjacencyList<int> cell_facets_max = mesh::get_entity_vertices(
      get_cell_type(max_vertices_per_cell, tdim), tdim - 1);
  int max_vertices_per_facet = 0;
  for (int i = 0; i < cell_facets_max.num_nodes(); ++i)
  {
    max_vertices_per_facet
        = std::max(max_vertices_per_facet, cell_facets_max.num_links(i));
  }

  const int shape1 = max_vertices_per_facet + 1;

  std::cout << "Test B: " << shape1 << std::endl;

  // Build list of facets, defined by sorted vertices, with connected
  // cell index at the end
  std::vector<std::int64_t> facets;
  facets.reserve(num_cells * cell_facets_max.num_nodes() * shape1);
  for (auto it = cell_offsets.begin(); it != std::prev(cell_offsets.end());
       ++it)
  {
    int num_vertices = *std::next(it) - *it;

    const graph::AdjacencyList<int> cell_facets = mesh::get_entity_vertices(
        get_cell_type(num_vertices, tdim), tdim - 1);

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

  std::vector<std::int64_t> unmatched_facets;
  std::vector<std::int32_t> cells;
  std::vector<std::array<std::int32_t, 2>> edges;
  {
    auto it = perm.begin();
    while (it != perm.end())
    {
      auto f0 = xtl::span(facets.data() + (*it) * shape1, shape1);

      // Find iterator to next facet different from f0
      auto it1 = std::find_if_not(
          it, perm.end(),
          [f0, &facets, shape1](auto idx) -> bool
          {
            std::size_t offset1 = idx * shape1;
            auto f1_it = std::next(facets.begin(), offset1);
            return std::equal(f0.begin(), std::prev(f0.end()), f1_it);
          });

      // Add dual graph edges (one direction only, other direction is
      // added later)
      std::int32_t cell0 = f0.back();
      for (auto itx = std::next(it); itx != it1; ++itx)
      {
        auto f1 = xtl::span(facets.data() + *itx * shape1, shape1);
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

  // Build adjacency list data

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
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>,
           std::size_t, std::vector<std::int32_t>>
mesh::build_local_dual_graph(const xtl::span<const std::int64_t>& cell_vertices,
                             const xtl::span<const std::int32_t>& cell_offsets,
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

  // enum class FacetType : int
  // {
  //   interval = 0,
  //   triangle = 1,
  //   quadrialteral = 2,
  //   tetrahedron = 3,
  //   pyramid = 4,
  //   prism = 5,
  //   pyramid = 6,
  // };

  int num_facets = 0;
  int max_num_facet_vertices = 0;
  switch (tdim)
  {
  case 1:
    if (num_cells_of_type[2] != num_cells)
      throw std::runtime_error("Invalid cells in 1D mesh");
    nv_to_facets[2] = mesh::get_entity_vertices(mesh::CellType::interval, 0);
    max_num_facet_vertices = 1;
    num_facets = 2 * num_cells_of_type[2];
    break;
  case 2:
    if (num_cells_of_type[3] + num_cells_of_type[4] != num_cells)
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
        != num_cells)
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
  std::vector<std::int32_t> facets(num_facets * max_num_facet_vertices,
                                   std::numeric_limits<std::int32_t>::max());
  std::vector<std::int32_t> facet_to_cell;
  facet_to_cell.reserve(num_facets);
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Cell facets (local) for current cell type
    const int num_cell_vertices = cell_offsets[c + 1] - cell_offsets[c];
    const graph::AdjacencyList<int>& cell_facets
        = nv_to_facets[num_cell_vertices];

    // Loop over all facets of cell c
    for (int f = 0; f < cell_facets.num_nodes(); ++f)
    {
      // Get data array for this facet
      xtl::span facet(facets.data()
                          + facet_to_cell.size() * max_num_facet_vertices,
                      max_num_facet_vertices);

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
  assert((int)facet_to_cell.size() == num_facets);

  // Sort facets by lexicographic order of vertices
  const std::size_t shape1 = max_num_facet_vertices;
  std::vector<int> facet_perm(facets.size() / max_num_facet_vertices);
  std::iota(facet_perm.begin(), facet_perm.end(), 0);
  std::sort(facet_perm.begin(), facet_perm.end(),
            [&facets, shape1](auto f0, auto f1)
            {
              auto it0 = std::next(facets.begin(), f0 * shape1);
              auto it1 = std::next(facets.begin(), f1 * shape1);
              return std::lexicographical_compare(it0, std::next(it0, shape1),
                                                  it1, std::next(it1, shape1));
            });

  // Iterator over facets, and push back cells that share the facet. If
  // facet is not shared, store in 'unshared_facets'.
  std::vector<std::int32_t> edges;
  edges.reserve(2 * num_cells);
  std::vector<std::int32_t> unshared_facets;
  unshared_facets.reserve(num_cells);
  int eq_count = 0;
  for (std::int32_t f = 1; f < num_facets; ++f)
  {
    xtl::span current(facets.data() + facet_perm[f] * max_num_facet_vertices,
                      max_num_facet_vertices);
    xtl::span previous(facets.data()
                           + facet_perm[f - 1] * max_num_facet_vertices,
                       max_num_facet_vertices);
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
  std::vector<std::int64_t> unmatched_facets(
      unshared_facets.size() * max_num_facet_vertices,
      std::numeric_limits<std::int64_t>::max());
  std::vector<std::int32_t> fcells;
  fcells.reserve(unshared_facets.size());
  for (auto f = unshared_facets.begin(); f != unshared_facets.end(); ++f)
  {
    std::size_t pos = std::distance(unshared_facets.begin(), f);
    xtl::span facet_unmatched(unmatched_facets.data()
                                  + pos * max_num_facet_vertices,
                              max_num_facet_vertices);
    xtl::span facet(facets.data() + (*f) * max_num_facet_vertices,
                    max_num_facet_vertices);
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
    // facet_unmatched.back() = facet_to_cell[*f];
    fcells.push_back(facet_to_cell[*f]);
  }

  // Count number of edges for each cell
  std::vector<std::int32_t> num_edges(num_cells, 0);
  for (std::int32_t cell : edges)
  {
    assert(cell < num_cells);
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
          std::move(unmatched_facets), max_num_facet_vertices,
          std::move(fcells)};
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int64_t>
mesh::build_dual_graph(const MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       int tdim)
{
  LOG(INFO) << "Build mesh dual graph";

  // Compute local part of dual graph (cells are graph nodes, and edges
  // are connections by facet)
  auto [xlocal_graph, xfacets, xshape1, xfcells]
      = mesh::build_local_dual_graph_new(cells.array(), cells.offsets(), tdim);
  auto [local_graph, facets, shape1, fcells]
      = mesh::build_local_dual_graph(cells.array(), cells.offsets(), tdim);

  assert(local_graph.num_nodes() == cells.num_nodes());
  std::cout << "Size check 0: " << xfcells.size() << ", " << fcells.size()
            << std::endl;

  std::cout << "Size check 1: " << xfacets.size() << ", " << facets.size()
            << std::endl;

  std::cout << "Size check 2: " << xshape1 << ", " << shape1 << std::endl;

  // assert(xfcells.size() == fcells.size());
  assert(xshape1 == shape1);

  // Extend with nonlocal edges and convert to global indices

  graph::AdjacencyList<std::int64_t> graph
      = compute_nonlocal_dual_graph(comm, facets, shape1, fcells, local_graph);

  LOG(INFO) << "Graph edges (local: " << local_graph.offsets().back()
            << ", non-local: "
            << graph.offsets().back() - local_graph.offsets().back() << ")";

  return graph;
}
//-----------------------------------------------------------------------------
