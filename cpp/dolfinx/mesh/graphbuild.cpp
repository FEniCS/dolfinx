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
#include <iostream>
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

  // Postoffice setup:
  //  a) facets need to globally decide on a consistent ownership model (without
  //  communication).
  //    - first (global) vertex index of a facet is used
  //    - dolfinx::MPI::index_owner deduces ownership
  //  b) every facet is send to owning PO
  //    - data for facet i: list of vertices + associated global cell idx
  //  c) check (on PO) if multiple same facets have been received
  //    - if so, found matched facet across process boundary -> introduce edge
  //      to dual graph
  //    - prepare info of matched facet for recipients
  //  d) return info of remotely matched/unmatched facets from PO
  //    - construct locally the parallel aware dual graph (with ghost edges).

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
        auto facet = facets.subspan(f * local_max_vertices_per_facet,
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

    // Build list of dest ranks and count number of items (facets) to
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
  // and pack into send buffer

  // TODO:
  // 1) change from sending back data in the format [-1, 10, 2, -1, ...] to
  // 'adjacency list based'.
  //      a) all_to_all the number of matched facets to send (per received
  //      facet) b) all_to_all list of all facet matches
  // 2) extend to multiple matched facets.

  std::vector<std::int64_t> send_buffer1(recv_disp.back(), -1);
  std::vector<std::vector<std::int64_t>> matched_facets(recv_disp.back());
  {
    // Compute sort permutation for received data
    std::vector<int> sort_order(recv_buffer.size() / buffer_shape1);
    std::iota(sort_order.begin(), sort_order.end(), 0);
    std::ranges::sort(sort_order, std::ranges::lexicographical_compare,
                      [&](auto f)
                      {
                        auto begin
                            = std::next(recv_buffer.begin(), f * buffer_shape1);
                        return std::ranges::subrange(
                            begin, std::next(begin, max_vertices_per_facet));
                      });

    for (auto it = sort_order.begin(); it != sort_order.end();)
    {
      std::size_t offset0 = (*it) * buffer_shape1;
      auto f0 = std::next(recv_buffer.data(), offset0);

      // Find iterator to next facet different from f0
      auto it1 = std::find_if_not(
          it, sort_order.end(),
          [f0, &recv_buffer, buffer_shape1,
           max_vertices_per_facet](auto idx) -> bool
          {
            std::size_t offset1 = idx * buffer_shape1;
            auto f1 = std::next(recv_buffer.data(), offset1);
            return std::equal(f0, std::next(f0, max_vertices_per_facet), f1);
          });

      // TODO: generalise for more than two matches and log warning
      // (maybe with an option?). Would need to send back multiple
      // values.
      if (std::size_t num_matches = std::distance(it, it1); num_matches == 2)
      {
        // Store the global cell index from the other rank
        int facet = *it;
        int next_facet = *(it + 1);
        send_buffer1[facet]
            = recv_buffer[next_facet * buffer_shape1 + max_vertices_per_facet];
        send_buffer1[next_facet]
            = recv_buffer[facet * buffer_shape1 + max_vertices_per_facet];
        matched_facets[facet].push_back(
            recv_buffer[next_facet * buffer_shape1 + max_vertices_per_facet]);
        matched_facets[next_facet].push_back(
            recv_buffer[facet * buffer_shape1 + max_vertices_per_facet]);
      }
      else if (num_matches > 2)
      {
        throw std::runtime_error(
            "A facet is connected to more than two cells.");
      }

      // Advance iterator and increment entity
      it = it1;
    }
  }

  std::vector<int> num_items_po_send;
  num_items_po_send.reserve(matched_facets.size());
  std::ranges::for_each(matched_facets, [&](const auto& matches)
                        { num_items_po_send.push_back(matches.size()); });

  // Create neighbourhood communicator for sending data from post
  // offices
  MPI_Comm comm_po_receive;
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &comm_po_receive);

  // std::vector<int> send_count_per_process; // TODO:
  // std::vector<int> send_displ_matched; // TODO

  // std::vector<std::int64_t> send_buffer2;
  // send_buffer2.reserve(
  //     std::accumulate(num_items_po_send.begin(), num_items_po_send.end(),
  //     0));
  // for (auto& matches : matched_facets)
  //   for (auto match : matches)
  //     send_buffer2.push_back(match);

  // // Send back matched facet data
  // std::vector<int> receive_count_per_process(src.size());
  // for (int i = 0; i < num_items_per_dest.size(); i++)
  // {
  //   int length = num_items_per_dest[i];
  //   auto begin = std::next(
  //       recv_matched_facet_counts.begin(),
  //       std::accumulate(num_items_per_dest.begin(),
  //                       std::next(num_items_per_dest.begin(), i), 0));
  //   receive_count_per_process[i]
  //       = std::accumulate(begin, std::next(begin, length), 0);
  // }

  // // Compute send displacements
  // std::vector<std::int32_t> send_disp_matched(
  //     receive_count_per_process.size() + 1, 0);
  // std::partial_sum(receive_count_per_process.begin(),
  //                  receive_count_per_process.end(),
  //                  std::next(send_disp_matched.begin()));

  // std::vector<std::int64_t> recv_matched_facets(std::accumulate(
  //     receive_count_per_process.begin(), receive_count_per_process.end(),
  //     0));
  // TODO: one of the displacements wrong?
  // MPI_Neighbor_alltoallv(
  //     send_buffer2.data(), send_count_per_process.data(),
  //     send_disp_matched.data(), dolfinx::MPI::mpi_t<std::int64_t>,
  //     recv_matched_facets.data(), receive_count_per_process.data(),
  //     send_disp_matched.data(), dolfinx::MPI::mpi_t<std::int64_t>,
  //     comm_po_receive);

  // Send back matched cell counts
  std::vector<int> recv_matched_facet_counts(send_disp.back());
  MPI_Neighbor_alltoallv(
      num_items_po_send.data(), num_items_recv.data(), recv_disp.data(),
      MPI_INT, recv_matched_facet_counts.data(), num_items_per_dest.data(),
      send_disp.data(), MPI_INT, comm_po_receive);

  // Send back data (TODO: remove once transition done)
  std::vector<std::int64_t> recv_buffer1(send_disp.back());
  MPI_Neighbor_alltoallv(send_buffer1.data(), num_items_recv.data(),
                         recv_disp.data(), dolfinx::MPI::mpi_t<std::int64_t>,
                         recv_buffer1.data(), num_items_per_dest.data(),
                         send_disp.data(), dolfinx::MPI::mpi_t<std::int64_t>,
                         comm_po_receive);
  MPI_Comm_free(&comm_po_receive);

  // Temporary check for recv_matched_facet_counts_aligns with recv_buffer1 -
  // received facets.
  for (int i = 0; i < recv_matched_facet_counts.size(); i++)
  {
    assert(i < recv_buffer1.size());
    auto matched_count = recv_matched_facet_counts[i];
    auto recv_index = recv_buffer1[i];
    std::cout << "matched_count: " << matched_count << "\n"
              << " recv_index: " << recv_index << std::endl;
    assert((matched_count == 1 and recv_index != -1) || (matched_count == 0));
  }

  // --- Build new graph

  // Count number of adjacency list edges
  std::vector<std::int32_t> num_edges(local_dual_graph.num_nodes(), 0);
  std::adjacent_difference(std::next(local_dual_graph.offsets().begin()),
                           local_dual_graph.offsets().end(), num_edges.begin());
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
  std::vector<std::int32_t> offsets(local_dual_graph.num_nodes() + 1, 0);
  std::partial_sum(num_edges.cbegin(), num_edges.cend(),
                   std::next(offsets.begin()));

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
      auto v = _cells.subspan(num_cell_vertices * c, num_cell_vertices);
      for (int f = 0; f < cell_facets.num_nodes(); ++f)
      {
        auto facet_vertices = cell_facets.links(f);
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
  // TODO: radix_sort? This is a heavy sort call.
  std::ranges::sort(perm, std::ranges::lexicographical_compare,
                    [&](auto f)
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
      // TODO: use a subrange here
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
  std::ranges::for_each(edges,
                        [&num_links](auto e)
                        {
                          ++num_links[e[0]];
                          ++num_links[e[1]];
                        });

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
