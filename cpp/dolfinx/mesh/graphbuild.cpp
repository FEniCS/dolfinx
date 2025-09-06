// Copyright (C) 2010-2025 Garth N. Wells and Paul T. Kühner
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
#include <mpi.h>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
/// @brief Build nonlocal part of dual graph for mesh and return number
/// of non-local edges.
///
/// @note Scalable version.
///
/// @note graphbuild::compute_local_dual_graph should be called
/// before this function is called.
///
/// @param[in] comm MPI communicator
/// @param[in] facets Facets on this rank that are shared by only on
/// cell on this rank, i.e. candidates for possibly residing on other
/// processes. Each row in `facets` corresponds to a facet, and the row
/// data has the form `[v0, ..., v_{n-1}, -1, -1]`, where `v_i` are the
/// sorted vertex global indices of the facets and `-1` is a padding
/// value for the mixed topology case where facets can have differing
/// number of vertices.
/// @param[in] local_max_vertices_per_facet Number of columns for `facets`.
/// @param[in] cells Attached cell (local index) for each facet in
/// `facet`.
/// @param[in] local_dual_graph The dual graph for cells on this MPI rank
///
/// @return Global dual graph, including ghost edges (edges to
/// off-procss cells)
graph::AdjacencyList<std::int64_t> compute_nonlocal_dual_graph(
    const MPI_Comm comm, std::span<const std::int64_t> facets,
    std::size_t local_max_vertices_per_facet,
    std::span<const std::int32_t> cells,
    const graph::AdjacencyList<std::int32_t>& local_dual_graph)
{
  spdlog::info("Build nonlocal part of mesh dual graph");
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

  const int comm_size = dolfinx::MPI::size(comm);

  // Return empty data if mesh is not distributed
  if (comm_size == 1)
  {
    // Convert graph to int64_t and return
    return graph::AdjacencyList(
        std::vector<std::int64_t>(local_dual_graph.array().begin(),
                                  local_dual_graph.array().end()),
        local_dual_graph.offsets());
  }

  // Postoffice (PO) setup:
  //  a) facets need to globally decide on a consistent ownership model (without
  //     communication).
  //    - first (global) vertex index of a facet is used
  //    - dolfinx::MPI::index_owner deduces ownership
  //  b) every unmatched facet is send to owning PO
  //    - data for every facet: list of vertices + associated global cell idx
  //  c) PO identifies ghost edges
  //    - PO checks if a facet has been received from multiple processes
  //    - If so, found matched facet across process boundary -> introduce edge
  //      to dual graph.
  //    - store for each received cell a list of (remote) matches.
  //  d) PO communicates matched cells back to senders
  //    - adjacencylist of facet to cell connectivity is communicated.
  //    - first the number of matched facets (link count)
  //    - the unrolled matched cells (data)
  //  e) combine local dual graph and remote edges into parallel aware dual
  //     graph.

  assert(local_max_vertices_per_facet == 0
         or facets.size() % local_max_vertices_per_facet == 0);
#ifndef NDEBUG
  {
    // assert facets sorted
    if (local_max_vertices_per_facet > 0)
    {
      for (std::size_t f = 0; f < facets.size() / local_max_vertices_per_facet;
           ++f)
      {
        std::span facet = facets.subspan(f * local_max_vertices_per_facet,
                                         local_max_vertices_per_facet);
        assert(std::is_sorted(facet.begin(), std::ranges::find(facet, -1)));
      }
    }
  }
#endif

  // Start (non-blocking) communication for cell offset
  std::int64_t cell_offset = 0;
  MPI_Request request_cell_offset;
  {
    const std::int64_t num_local = local_dual_graph.num_nodes();
    MPI_Iexscan(&num_local, &cell_offset, 1, MPI_INT64_T, MPI_SUM, comm,
                &request_cell_offset);
  }

  // Compute max_vertices_per_facet and vertex_range =
  // [min_vertex_index, max_vertex_index] across all processes. Use
  // first facet vertex for min/max index.
  std::int32_t max_vertices_per_facet = -1;
  std::array<std::int64_t, 2> vertex_range;
  {
    // Compute local quantities.
    // Note: to allow for single reduction we store -min_vertex_index,
    // i.e. max (-min_vertex_index) = min (min_vertex_index).
    max_vertices_per_facet = std::int64_t(local_max_vertices_per_facet);

    vertex_range[0] = std::numeric_limits<std::int64_t>::min();
    vertex_range[1] = -1;

    for (std::size_t i = 0; i < facets.size();
         i += local_max_vertices_per_facet)
    {
      vertex_range[0] = std::max(vertex_range[0], -facets[i]);
      vertex_range[1] = std::max(vertex_range[1], facets[i]);
    }

    // Exchange.
    std::array<std::int64_t, 3> send
        = {max_vertices_per_facet, vertex_range[0], vertex_range[1]};
    std::array<std::int64_t, 3> recv;
    MPI_Allreduce(send.data(), recv.data(), 3, MPI_INT64_T, MPI_MAX, comm);
    assert(recv[1] != std::numeric_limits<std::int64_t>::min());
    assert(recv[2] != -1);

    // Unpack.
    max_vertices_per_facet = recv[0];
    vertex_range = {-recv[1], recv[2] + 1};
  }
  spdlog::debug("Max. vertices per facet={}", max_vertices_per_facet);
  const std::int32_t buffer_shape1 = max_vertices_per_facet + 1;

  // Build list of dest ranks and count number of items (facets) to send
  // to each dest post office (by neighbourhood rank)
  const std::size_t facet_count = cells.size();
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest,
      pos_to_neigh_rank(facet_count, -1);
  {
    // Build {dest, pos} list for each facet, and sort (dest is the post
    // office rank)
    std::vector<std::array<std::int32_t, 2>> dest_to_index;
    dest_to_index.reserve(facet_count);
    std::int64_t range = vertex_range[1] - vertex_range[0];
    for (std::size_t f = 0; f < facet_count; ++f)
    {
      std::int64_t v0
          = facets[f * local_max_vertices_per_facet] - vertex_range[0];
      dest_to_index.push_back({dolfinx::MPI::index_owner(comm_size, v0, range),
                               static_cast<int>(f)});
    }
    std::ranges::sort(dest_to_index);

    // Build list of dest ranks and count number of items (facets+cell) to
    // send to each dest post office (by neighbourhood rank)
    for (auto it = dest_to_index.begin(); it != dest_to_index.end();)
    {
      const int neigh_rank = dest.size();

      // Store global rank
      dest.push_back(it->front());

      // Find iterator to next global rank
      auto it1
          = std::find_if(it, dest_to_index.end(),
                         [r = dest.back()](auto& idx) { return idx[0] != r; });

      // Store number of items for current rank
      num_items_per_dest.push_back(std::distance(it, it1));

      // Set entry in map from local facet row index (position) to local
      // destination rank
      for (auto& e : std::ranges::subrange(it, it1))
        pos_to_neigh_rank[e[1]] = neigh_rank;

      // Advance iterator
      it = it1;
    }
  }

  assert(num_items_per_dest.size() == dest.size());

  // Determine source ranks
  const std::vector<int> src
      = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  spdlog::info("Number of destination and source ranks in non-local dual graph "
               "construction, and ratio to total number of ranks: {}, {}, "
               "{}, {}",
               dest.size(), src.size(),
               static_cast<double>(dest.size()) / comm_size,
               static_cast<double>(src.size()) / comm_size);

  // Create neighbourhood communicator for sending data to
  // post offices
  MPI_Comm comm_po_post;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &comm_po_post);

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
    for (std::size_t f = 0; f < facet_count; ++f)
    {
      int neigh_dest = pos_to_neigh_rank[f];
      std::size_t pos = send_offsets[neigh_dest];
      send_indx_to_pos[pos] = f;

      // Copy facet data into buffer
      auto fdata = facets.subspan(f * local_max_vertices_per_facet,
                                  local_max_vertices_per_facet);
      std::span send_buffer_f(send_buffer.data() + buffer_shape1 * pos,
                              max_vertices_per_facet + 1);
      std::ranges::copy(fdata, send_buffer_f.begin());
      send_buffer_f.back() = cells[f] + cell_offset;
      ++send_offsets[neigh_dest];
    }
  }

  // Send number of send items to post offices
  std::vector<int> num_items_recv(src.size());
  num_items_per_dest.reserve(1);
  num_items_recv.reserve(1);
  MPI_Neighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                        num_items_recv.data(), 1, MPI_INT, comm_po_post);

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
                         comm_po_post);

  MPI_Type_free(&compound_type);
  MPI_Comm_free(&comm_po_post);

  // Search for consecutive facets (-> dual graph edge between cells)
  // and pack into send buffer. We store for every cell the number of matches,
  // the offsets of each cell and the continuous data.
  // Note: deges is short for dual edges.
  std::vector<int> dedge_send_count(recv_disp.back());
  std::vector<std::int32_t> dedge_send_displs(dedge_send_count.size() + 1, 0);
  std::vector<std::int64_t> dedge_send_data;
  {
    // Compute sort permutation for received data
    std::vector<int> sort_order(recv_buffer.size() / buffer_shape1);
    std::iota(sort_order.begin(), sort_order.end(), 0);
    std::ranges::sort(
        sort_order, std::ranges::lexicographical_compare,
        [max_vertices_per_facet, buffer_shape1, &recv_buffer](auto f)
        {
          auto begin = std::next(recv_buffer.begin(), f * buffer_shape1);
          return std::ranges::subrange(
              begin, std::next(begin, max_vertices_per_facet));
        });

    auto for_each_matched_pair = [buffer_shape1, max_vertices_per_facet,
                                  &sort_order, &recv_buffer](auto&& lambda)
    {
      for (auto it = sort_order.begin(); it != sort_order.end();)
      {
        std::size_t offset0 = (*it) * buffer_shape1;
        auto f0 = std::next(recv_buffer.data(), offset0);

        // Find range of equal facets f0.
        auto matching_facets = std::ranges::subrange(
            it, std::find_if_not(
                    it, sort_order.end(),
                    [f0, &recv_buffer, buffer_shape1,
                     max_vertices_per_facet](auto idx) -> bool
                    {
                      std::size_t offset1 = idx * buffer_shape1;
                      auto f1 = std::next(recv_buffer.data(), offset1);
                      return std::equal(
                          f0, std::next(f0, max_vertices_per_facet), f1);
                    }));

        for (auto facet_a_it = matching_facets.begin();
             facet_a_it != matching_facets.end(); facet_a_it++)
        {
          for (auto facet_b_it = std::next(facet_a_it);
               facet_b_it != matching_facets.end(); facet_b_it++)
          {
            int facet_a = *facet_a_it;
            int facet_b = *facet_b_it;

            std::int64_t cell_a
                = recv_buffer[facet_a * buffer_shape1 + max_vertices_per_facet];
            std::int64_t cell_b
                = recv_buffer[facet_b * buffer_shape1 + max_vertices_per_facet];

            lambda(facet_a, cell_a, facet_b, cell_b);
          }
        }
        it = matching_facets.end();
      }
    };

    // Iterate matching facets to compute count/offset information of dual edges
    for_each_matched_pair(
        [&dedge_send_count](int facet_a, std::int64_t /* cell_a */, int facet_b,
                            std::int64_t /* cell_b */)
        {
          ++dedge_send_count[facet_a];
          ++dedge_send_count[facet_b];
        });

    std::partial_sum(dedge_send_count.begin(), dedge_send_count.end(),
                     std::next(dedge_send_displs.begin()));

    std::int32_t send_dual_edges_size
        = std::accumulate(dedge_send_count.begin(), dedge_send_count.end(), 0);
    dedge_send_data.resize(send_dual_edges_size);

    // Iterate matching facets to store dual edges
    std::vector<std::int32_t> offset = dedge_send_displs;
    for_each_matched_pair(
        [&dedge_send_data, &offset](int facet_a, std::int64_t cell_a,
                                    int facet_b, std::int64_t cell_b)
        {
          dedge_send_data[offset[facet_a]++] = cell_b;
          dedge_send_data[offset[facet_b]++] = cell_a;
        });
  }

  // Create neighbourhood communicator for sending data from post
  // offices
  MPI_Comm comm_po_receive;
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &comm_po_receive);

  // Send PO->recipient: matched cell counts (non-blocking)
  std::vector<int> dedge_recv_count(send_disp.back());
  MPI_Request dedge_recv_count_request;
  MPI_Ineighbor_alltoallv(dedge_send_count.data(), num_items_recv.data(),
                          recv_disp.data(), MPI_INT, dedge_recv_count.data(),
                          num_items_per_dest.data(), send_disp.data(), MPI_INT,
                          comm_po_receive, &dedge_recv_count_request);

  // Prepare send data for matched facets. Note, we have prepared adjacency
  // information for all cells. Here we retrieve the offset and displacement
  // data corresponding to the per process adjacencylists.
  // Note: pp in variable names is short for per-process.
  std::vector<int> dedge_send_count_pp(num_items_recv.size(), 0);
  std::vector<std::int32_t> dedge_send_displs_pp(dedge_send_count_pp.size() + 1,
                                                 0);
  {
    int index = 0;
    for (std::size_t i = 0; i < num_items_recv.size(); i++)
    {
      for (int j = 0; j < num_items_recv[i]; j++)
        dedge_send_count_pp[i] += dedge_send_count[index + j];

      index += num_items_recv[i];
    }

    std::partial_sum(dedge_send_count_pp.begin(), dedge_send_count_pp.end(),
                     std::next(dedge_send_displs_pp.begin()));
  }

  // Compute matched facet receive counts and displacements.
  std::vector<int> dedge_recv_count_pp(num_items_per_dest.size(), 0);
  std::vector<std::int32_t> dedge_recv_displs_pp(dedge_recv_count_pp.size() + 1,
                                                 0);
  MPI_Wait(&dedge_recv_count_request, MPI_STATUS_IGNORE);
  {
    int index = 0;
    for (std::size_t i = 0; i < num_items_per_dest.size(); i++)
    {
      for (int j = 0; j < num_items_per_dest[i]; j++)
        dedge_recv_count_pp[i] += dedge_recv_count[index + j];

      index += num_items_per_dest[i];
    }

    std::partial_sum(dedge_recv_count_pp.begin(), dedge_recv_count_pp.end(),
                     std::next(dedge_recv_displs_pp.begin()));
  }
  // Exchange flattened list of matched facets
  std::vector<std::int64_t> recv_dual_edges(dedge_recv_displs_pp.back());
  MPI_Neighbor_alltoallv(dedge_send_data.data(), dedge_send_count_pp.data(),
                         dedge_send_displs_pp.data(),
                         dolfinx::MPI::mpi_t<std::int64_t>,
                         recv_dual_edges.data(), dedge_recv_count_pp.data(),
                         dedge_recv_displs_pp.data(),
                         dolfinx::MPI::mpi_t<std::int64_t>, comm_po_receive);

  MPI_Comm_free(&comm_po_receive);

  // --- Build global dual graph

  // Compute adjacency list offsets
  std::vector<std::int32_t> offsets(local_dual_graph.num_nodes() + 1, 0);
  {
    // Count number of adjacency list edges
    std::vector<std::int32_t> num_edges(local_dual_graph.num_nodes(), 0);
    std::adjacent_difference(std::next(local_dual_graph.offsets().begin()),
                             local_dual_graph.offsets().end(),
                             num_edges.begin());

    for (std::size_t i = 0; i < dedge_recv_count.size(); ++i)
    {
      std::size_t cell_idx = send_indx_to_pos[i];
      std::size_t cell = cells[cell_idx];
      num_edges[cell] += dedge_recv_count[i];
    }

    // Compute adjacency list offsets
    std::partial_sum(num_edges.cbegin(), num_edges.cend(),
                     std::next(offsets.begin()));
  }

  // Compute adjacency list data (edges)
  std::vector<std::int64_t> data(offsets.back());
  {
    std::vector<std::int32_t> disp = offsets;

    // Copy local data and add cell offset
    for (std::int32_t i = 0; i < local_dual_graph.num_nodes(); ++i)
    {
      auto e = local_dual_graph.links(i);
      disp[i] += e.size();
      std::ranges::transform(e, std::next(data.begin(), offsets[i]),
                             [cell_offset](auto x) { return x + cell_offset; });
    }

    // Add non-local data
    int offset = 0;
    for (std::size_t i = 0; i < dedge_recv_count.size(); i++)
    {
      std::int32_t cell_idx = send_indx_to_pos[i];
      std::int32_t cell = cells[cell_idx];

      for (int j = 0; j < dedge_recv_count[i]; j++)
      {
        std::int32_t _cell_offset = disp[cell]++;
        std::int64_t node = recv_dual_edges[offset + j];
        data[_cell_offset] = node;
      }

      offset += dedge_recv_count[i];
    }
    // local connections are possibly introduced again by remote -> remove
    // duplicates
    std::size_t duplicates_count = 0;
    for (std::size_t node = 0; node < offsets.size() - 1; node++)
    {
      // Account for offset
      offsets[node] -= duplicates_count;

      auto links
          = std::ranges::subrange(std::next(data.begin(), offsets[node]),
                                  std::next(data.begin(), offsets[node + 1]));
      std::ranges::sort(links);
      auto duplicate_links = std::ranges::unique(links);
      if (duplicate_links.empty())
        continue;

      data.erase(duplicate_links.begin(), duplicate_links.end());
      duplicates_count += std::ranges::size(duplicate_links);
    }
    offsets[offsets.size() - 1] -= duplicates_count;
  }

  return graph::AdjacencyList(std::move(data), std::move(offsets));
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>,
           std::size_t, std::vector<std::int32_t>>
mesh::build_local_dual_graph(
    std::span<const CellType> celltypes,
    const std::vector<std::span<const std::int64_t>>& cells,
    std::optional<std::int32_t> max_facet_to_cell_links)
{
  spdlog::info("Build local part of mesh dual graph (mixed)");
  common::Timer timer("Compute local part of mesh dual graph (mixed)");

  if (std::size_t ncells_local
      = std::accumulate(cells.begin(), cells.end(), 0,
                        [](std::size_t s, std::span<const std::int64_t> c)
                        { return s + c.size(); });
      ncells_local == 0)
  {
    // Empty mesh on this process
    return {graph::AdjacencyList<std::int32_t>(0), std::vector<std::int64_t>(),
            0, std::vector<std::int32_t>()};
  }

  if (cells.size() != celltypes.size())
  {
    throw std::runtime_error(
        "Number of cell types must match number of cell arrays.");
  };

  int tdim = mesh::cell_dim(celltypes.front());

  // 1) Create indexing offset for each cell type and determine max
  //    number of vertices per facet -> size computations for later on
  //    used data structures

  // TODO: cell_offsets can be removed?
  std::vector<std::int32_t> cell_offsets{0};
  cell_offsets.reserve(cells.size() + 1);

  int max_vertices_per_facet = 0;
  int facet_count = 0;
  for (std::size_t j = 0; j < cells.size(); ++j)
  {
    CellType cell_type = celltypes[j];
    std::span<const std::int64_t> _cells = cells[j];

    assert(tdim == mesh::cell_dim(cell_type));

    int num_cell_vertices = mesh::cell_num_entities(cell_type, 0);
    int num_cell_facets = mesh::cell_num_entities(cell_type, tdim - 1);

    std::int32_t num_cells = _cells.size() / num_cell_vertices;
    cell_offsets.push_back(cell_offsets.back() + num_cells);
    facet_count += num_cell_facets * num_cells;

    graph::AdjacencyList<std::int32_t> cell_facets
        = mesh::get_entity_vertices(cell_type, tdim - 1);

    // Determine/update maximum number of vertices for facet
    std::ranges::for_each(
        std::views::iota(0, cell_facets.num_nodes()),
        [&max = max_vertices_per_facet, &cell_facets](auto node)
        { max = std::max(max, cell_facets.num_links(node)); });
  }

  // 2) Build a list of (all) facets, defined by sorted vertices, with
  //    the connected cell index after the vertices. For v_ij the j-th
  //    vertex of the i-th facet. The last index is the cell index (non
  //    unique).
  // facets = [v_11, v_12, v_13, -1, ..., -1, 0,
  //           v_21, v_22, v_23, -1, ..., -1, 0,
  //             ⋮     ⋮      ⋮    ⋮   ⋱    ⋮  ⋮
  //           v_n1, v_n2,   -1, -1, ..., -1, n]

  const int shape1 = max_vertices_per_facet + 1;
  std::vector<std::int64_t> facets;
  facets.reserve(facet_count * shape1);
  constexpr std::int32_t padding_value = -1;

  for (std::size_t j = 0; j < cells.size(); ++j)
  {
    const CellType& cell_type = celltypes[j];
    std::span _cells = cells[j];

    int num_cell_vertices = mesh::cell_num_entities(cell_type, 0);
    std::int32_t num_cells = _cells.size() / num_cell_vertices;
    graph::AdjacencyList<int> cell_facets
        = mesh::get_entity_vertices(cell_type, tdim - 1);

    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      // Loop over cell facets
      std::span v = _cells.subspan(num_cell_vertices * c, num_cell_vertices);
      for (int f = 0; f < cell_facets.num_nodes(); ++f)
      {
        std::span facet_vertices = cell_facets.links(f);
        std::ranges::transform(facet_vertices, std::back_inserter(facets),
                               [v](auto idx) { return v[idx]; });
        // TODO: radix_sort?
        std::sort(std::prev(facets.end(), facet_vertices.size()), facets.end());
        facets.insert(facets.end(),
                      max_vertices_per_facet - facet_vertices.size(),
                      padding_value);
        facets.push_back(c + cell_offsets[j]);
      }
    }
  }

  // 3) Sort facets by vertex key
  std::vector<std::size_t> perm(facets.size() / shape1, 0);
  std::iota(perm.begin(), perm.end(), 0);
  std::ranges::sort(perm, std::ranges::lexicographical_compare,
                    [&facets, shape1](auto f)
                    {
                      auto begin = std::next(facets.begin(), f * shape1);
                      return std::ranges::subrange(begin,
                                                   std::next(begin, shape1));
                    });

  // // 4) Iterate over sorted list of facets. Facets shared by more than
  //    one cell lead to a graph edge to be added. Facets that are not
  //    shared are stored as these might be shared by a cell on another
  //    process.
  std::vector<std::int64_t> unmatched_facets;
  std::vector<std::int32_t> local_cells;
  std::vector<std::array<std::int32_t, 2>> edges;
  {
    for (auto it = perm.begin(); it != perm.end();)
    {
      std::size_t facet_index = *it;
      std::span facet(facets.data() + facet_index * shape1, shape1);

      // Find iterator to next facet different from f0 -> all facets in
      // [it, it_next_facet) describe the same facet
      auto matching_facets = std::ranges::subrange(
          it, std::find_if_not(it, perm.end(),
                               [facet, &facets, shape1](auto idx) -> bool
                               {
                                 auto f1_it
                                     = std::next(facets.begin(), idx * shape1);
                                 return std::equal(facet.begin(),
                                                   std::prev(facet.end()),
                                                   f1_it);
                               }));

      std::int32_t cell_count = matching_facets.size();
      assert(cell_count >= 1);
      if (!max_facet_to_cell_links.has_value()
          or (cell_count < *max_facet_to_cell_links))
      {
        // Store unmatched facets and the attached cell
        for (std::int32_t i = 0; i < cell_count; i++)
        {
          unmatched_facets.insert(unmatched_facets.end(), facet.begin(),
                                  std::prev(facet.end()));
          std::int32_t cell = facets[*std::next(it, i) * shape1 + (shape1 - 1)];
          local_cells.push_back(cell);
        }
      }

      // Add dual graph edges (one direction only, other direction is
      // added later). In the range [it, it_next_facet), all
      // combinations are added.
      for (auto facet_a_it = it; facet_a_it != matching_facets.end();
           facet_a_it++)
      {
        std::span facet_a(facets.data() + *facet_a_it * shape1, shape1);
        std::int32_t cell_a = facet_a.back();
        for (auto facet_b_it = std::next(facet_a_it);
             facet_b_it != matching_facets.end(); facet_b_it++)
        {
          std::span facet_b(facets.data() + *facet_b_it * shape1, shape1);
          std::int32_t cell_b = facet_b.back();
          edges.push_back({cell_a, cell_b});
        }
      }

      // Update iterator
      it = matching_facets.end();
    }
  }

  // 5) Build adjacency list data. Prepare data structure and assemble
  //    into. Important: we have only computed one direction of the dual
  //    edges, we add both forward and backward to the final data
  //    structure.

  std::vector<std::int32_t> num_links(cell_offsets.back(), 0);

  for (auto [a, b] : edges)
  {
    ++num_links[a];
    ++num_links[b];
  }

  std::vector<std::int32_t> offsets(num_links.size() + 1, 0);
  std::partial_sum(num_links.cbegin(), num_links.cend(),
                   std::next(offsets.begin()));
  std::vector<std::int32_t> data(offsets.back());
  std::ranges::for_each(edges,
                        [&data, pos = offsets](auto e) mutable
                        {
                          data[pos[e[0]]++] = e[1];
                          data[pos[e[1]]++] = e[0];
                        });

  return {graph::AdjacencyList(std::move(data), std::move(offsets)),
          std::move(unmatched_facets), max_vertices_per_facet,
          std::move(local_cells)};
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int64_t>
mesh::build_dual_graph(MPI_Comm comm, std::span<const CellType> celltypes,
                       const std::vector<std::span<const std::int64_t>>& cells,
                       std::optional<std::int32_t> max_facet_to_cell_links)
{
  spdlog::info("Building mesh dual graph");

  // Compute local part of dual graph (cells are graph nodes, and edges
  // are connections by facet)
  auto [local_graph, facets, shape1, fcells]
      = mesh::build_local_dual_graph(celltypes, cells, max_facet_to_cell_links);

  // Extend with nonlocal edges and convert to global indices
  graph::AdjacencyList graph
      = compute_nonlocal_dual_graph(comm, facets, shape1, fcells, local_graph);

  spdlog::info("Graph edges (local: {}, non-local: {})",
               local_graph.offsets().back(),
               graph.offsets().back() - local_graph.offsets().back());

  return graph;
}
//-----------------------------------------------------------------------------
