// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <format>
#include <iterator>
#include <string>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::tuple<int, std::size_t, std::int8_t>,
                     std::pair<std::int32_t, std::int32_t>>
graph::comm_graph(const common::IndexMap& map, int root)
{
  MPI_Comm comm = map.comm();

  std::span<const int> dest = map.dest();
  int ierr;

  // Graph edge out(dest) weights
  const std::vector<std::int32_t> w_dest = map.weights_dest();

  // Group ranks by type
  const auto [local_dest, local_src] = map.rank_type(MPI_COMM_TYPE_SHARED);

  // Get number of edges for each node (rank)
  int num_edges_local = dest.size();
  std::vector<int> num_edges_remote(dolfinx::MPI::size(comm));
  ierr = MPI_Gather(&num_edges_local, 1, MPI_INT, num_edges_remote.data(), 1,
                    MPI_INT, root, comm);
  dolfinx::MPI::check_error(comm, ierr);

  // Compute displacements
  std::vector<int> disp(num_edges_remote.size() + 1, 0);
  std::partial_sum(num_edges_remote.begin(), num_edges_remote.end(),
                   std::next(disp.begin()));
  dolfinx::MPI::check_error(comm, ierr);

  // For each node (rank), get edge indices
  std::vector<int> edges_remote(disp.back());
  edges_remote.reserve(1);
  ierr = MPI_Gatherv(dest.data(), dest.size(), MPI_INT, edges_remote.data(),
                     num_edges_remote.data(), disp.data(), MPI_INT, root, comm);
  dolfinx::MPI::check_error(comm, ierr);

  // For each edge, get edge weight
  std::vector<std::int32_t> weights_remote(disp.back());
  weights_remote.reserve(1);
  ierr = MPI_Gatherv(w_dest.data(), w_dest.size(), MPI_INT32_T,
                     weights_remote.data(), num_edges_remote.data(),
                     disp.data(), MPI_INT32_T, root, comm);
  dolfinx::MPI::check_error(comm, ierr);

  // For node get local and ghost sizes
  std::vector<std::pair<std::int32_t, std::int32_t>> sizes_remote;
  {
    std::vector<std::int32_t> sizes_local(dolfinx::MPI::size(comm));
    std::int32_t size = map.size_local();
    ierr = MPI_Gather(&size, 1, MPI_INT32_T, sizes_local.data(), 1, MPI_INT32_T,
                      root, comm);
    dolfinx::MPI::check_error(comm, ierr);

    std::vector<std::int32_t> sizes_ghost(dolfinx::MPI::size(comm));
    std::int32_t num_ghosts = map.num_ghosts();
    ierr = MPI_Gather(&num_ghosts, 1, MPI_INT32_T, sizes_ghost.data(), 1,
                      MPI_INT32_T, root, comm);
    dolfinx::MPI::check_error(comm, ierr);

    std::transform(sizes_local.begin(), sizes_local.end(), sizes_ghost.begin(),
                   std::back_inserter(sizes_remote),
                   [](auto x, auto y) { return std::pair(x, y); });
  }

  // For each edge, get its local/remote marker
  std::vector<std::int8_t> markers;
  for (auto r : dest)
  {
    auto it = std::ranges::lower_bound(local_dest, r);
    if (it != local_dest.end() and *it == r)
      markers.push_back(1);
    else
      markers.push_back(0);
  }
  std::vector<std::int8_t> markers_remote(disp.back());
  ierr = MPI_Gatherv(markers.data(), markers.size(), MPI_INT8_T,
                     markers_remote.data(), num_edges_remote.data(),
                     disp.data(), MPI_INT8_T, root, comm);
  dolfinx::MPI::check_error(comm, ierr);

  std::vector<std::tuple<int, std::size_t, std::int8_t>> e_data;
  for (std::size_t i = 0; i < edges_remote.size(); ++i)
    e_data.emplace_back(edges_remote[i], weights_remote[i], markers_remote[i]);
  return graph::AdjacencyList(std::move(e_data),
                              std::vector(disp.begin(), disp.end()),
                              std::move(sizes_remote));
}
//-----------------------------------------------------------------------------
std::string graph::comm_to_json(
    const graph::AdjacencyList<std::tuple<int, std::size_t, std::int8_t>,
                               std::pair<std::int32_t, std::int32_t>>& g)
{
  const std::vector<std::pair<std::int32_t, std::int32_t>>& node_weights
      = g.node_data().value();

  std::string out
      = R"({"directed": true, "multigraph": false, "graph": [], "nodes": [)";
  for (std::int32_t n = 0; n < g.num_nodes(); ++n)
  {
    // Note: it is helpful to order map keys alphabetically
    std::format_to(std::back_inserter(out),
                   R"({{"num_ghosts": {}, "weight": {},  "id": {}}})",
                   node_weights[n].second, node_weights[n].first, n);
    if (n != g.num_nodes() - 1)
      out += ", ";
  }
  out += R"(], "adjacency": [)";
  for (std::int32_t n = 0; n < g.num_nodes(); ++n)
  {
    out += "[";
    auto links = g.links(n);
    for (std::size_t edge = 0; edge < links.size(); ++edge)
    {
      auto [e, w, local] = links[edge];
      std::format_to(std::back_inserter(out),
                     R"({{"local": {}, "weight": {}, "id": {}}})", local, w, e);
      if (edge != links.size() - 1)
        out += ", ";
    }
    out += "]";
    if (n != g.num_nodes() - 1)
      out += ", ";
  }
  out += "]}";

  return out;
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<int>
graph::index_to_dest_ranks(const common::IndexMap& map, int tag)
{
  MPI_Comm comm = map.comm();
  const std::int64_t offset = map.local_range()[0];

  // Build lists of src and dest ranks
  std::vector<int> src(map.owners().begin(), map.owners().end());
  std::ranges::sort(src);
  auto [unique_end, range_end] = std::ranges::unique(src);
  src.erase(unique_end, range_end);
  std::vector<int> dest = dolfinx::MPI::compute_graph_edges_nbx(comm, src, tag);
  std::ranges::sort(dest);

  // Array (local idx, ghosting rank) pairs for owned indices
  std::vector<std::pair<std::int32_t, int>> idx_to_rank;

  // 1. Build adjacency list data for owned indices (index, [sharing
  //    ranks])
  std::vector<std::int32_t> offsets{0};
  std::vector<int> data;
  {
    // Build list of (owner rank, index) pairs for each ghost index, and sort
    std::vector<std::pair<int, std::int64_t>> owner_to_ghost;
    std::ranges::transform(map.ghosts(), map.owners(),
                           std::back_inserter(owner_to_ghost),
                           [](auto idx, auto r) -> std::pair<int, std::int64_t>
                           { return {r, idx}; });
    std::ranges::sort(owner_to_ghost);

    // Build send buffer (the second component of each pair in
    // owner_to_ghost) to send to rank that owns the index
    std::vector<std::int64_t> send_buffer;
    send_buffer.reserve(owner_to_ghost.size());
    std::ranges::transform(owner_to_ghost, std::back_inserter(send_buffer),
                           [](auto x) { return x.second; });

    // Compute send sizes and displacements
    std::vector<int> send_sizes, send_disp{0};
    auto it = owner_to_ghost.begin();
    while (it != owner_to_ghost.end())
    {
      auto it1 = std::find_if(it, owner_to_ghost.end(),
                              [r = it->first](auto x) { return x.first != r; });
      send_sizes.push_back(std::distance(it, it1));
      send_disp.push_back(send_disp.back() + send_sizes.back());
      it = it1;
    }

    // Create ghost -> owner comm
    MPI_Comm comm0;
    int ierr = MPI_Dist_graph_create_adjacent(
        comm, dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(), src.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    dolfinx::MPI::check_error(comm, ierr);

    // Exchange number of indices to send/receive from each rank
    std::vector<int> recv_sizes(dest.size(), 0);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT,
                                 recv_sizes.data(), 1, MPI_INT, comm0);
    dolfinx::MPI::check_error(comm, ierr);

    // Prepare receive displacement array
    std::vector<int> recv_disp(dest.size() + 1, 0);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::next(recv_disp.begin()));

    // Send ghost indices to owner, and receive owned indices
    std::vector<std::int64_t> recv_buffer(recv_disp.back());
    ierr = MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                                  send_disp.data(), MPI_INT64_T,
                                  recv_buffer.data(), recv_sizes.data(),
                                  recv_disp.data(), MPI_INT64_T, comm0);
    dolfinx::MPI::check_error(comm, ierr);
    ierr = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(comm, ierr);

    // Build array of (local index, ghosting local rank), and sort
    for (std::size_t r = 0; r < recv_disp.size() - 1; ++r)
    {
      for (int j = recv_disp[r]; j < recv_disp[r + 1]; ++j)
      {
        idx_to_rank.push_back(
            {static_cast<std::int32_t>(recv_buffer[j] - offset),
             static_cast<int>(r)});
      }
    }
    std::ranges::sort(idx_to_rank);

    // -- Send to ranks that ghost my indices all the sharing ranks

    // Build adjacency list data for (owned index) -> (ghosting ranks)
    data.reserve(idx_to_rank.size());
    std::ranges::transform(idx_to_rank, std::back_inserter(data),
                           [](auto x) { return x.second; });
    offsets.reserve(map.size_local() + map.num_ghosts() + 1);
    {
      auto it = idx_to_rank.begin();

      // Loop over owned indices
      for (std::int32_t i = 0; i < map.size_local(); ++i)
      {
        auto it1 = std::find_if(it, idx_to_rank.end(),
                                [i](auto x) { return x.first != i; });
        offsets.push_back(offsets.back() + std::distance(it, it1));
        it = it1;
      }
    }
  }

  // 2. Build and add adjacency list data for non-owned indices
  //    (index, [sharing ranks]). Non-owned indices are ghosted but
  //    not owned by this rank.
  {
    // Send data for owned indices back to ghosting ranks (this is
    // necessary to share with ghosting ranks all the ranks that also
    // ghost a ghost index)
    std::vector<std::int64_t> send_buffer;
    std::vector<int> send_sizes;
    {
      const int rank = dolfinx::MPI::rank(comm);
      std::vector<std::vector<std::int64_t>> dest_idx_to_rank(dest.size());
      for (std::size_t n = 0; n < offsets.size() - 1; ++n)
      {
        std::span<const std::int32_t> ranks(data.data() + offsets[n],
                                            offsets[n + 1] - offsets[n]);
        for (auto r0 : ranks)
        {
          for (auto r : ranks)
          {
            assert(r0 < static_cast<int>(dest_idx_to_rank.size()));
            if (r0 != r)
            {
              dest_idx_to_rank[r0].push_back(n + offset);
              dest_idx_to_rank[r0].push_back(dest[r]);
            }
          }
          dest_idx_to_rank[r0].push_back(n + offset);
          dest_idx_to_rank[r0].push_back(rank);
        }
      }

      // Count number of ghosts per destination and build send buffer
      std::ranges::transform(dest_idx_to_rank, std::back_inserter(send_sizes),
                             [](auto& x) -> int { return x.size(); });
      for (auto& d : dest_idx_to_rank)
        send_buffer.insert(send_buffer.end(), d.begin(), d.end());

      // Create owner -> ghost comm
      MPI_Comm comm0;
      int ierr = MPI_Dist_graph_create_adjacent(
          comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
          dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
      dolfinx::MPI::check_error(comm, ierr);

      // Send how many indices I ghost to each owner, and receive how
      // many of my indices other ranks ghost
      std::vector<int> recv_sizes(src.size(), 0);
      send_sizes.reserve(1);
      recv_sizes.reserve(1);
      ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT,
                                   recv_sizes.data(), 1, MPI_INT, comm0);
      dolfinx::MPI::check_error(comm, ierr);

      // Prepare displacement vectors
      std::vector<int> send_disp(dest.size() + 1, 0);
      std::vector<int> recv_disp(src.size() + 1, 0);
      std::partial_sum(send_sizes.begin(), send_sizes.end(),
                       std::next(send_disp.begin()));
      std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                       std::next(recv_disp.begin()));

      std::vector<std::int64_t> recv_indices(recv_disp.back());
      ierr = MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                                    send_disp.data(), MPI_INT64_T,
                                    recv_indices.data(), recv_sizes.data(),
                                    recv_disp.data(), MPI_INT64_T, comm0);
      dolfinx::MPI::check_error(comm, ierr);
      ierr = MPI_Comm_free(&comm0);
      dolfinx::MPI::check_error(comm, ierr);

      // Build list of (ghost index, ghost position) pairs for indices
      // ghosted by this rank, and sort
      std::vector<std::pair<std::int64_t, std::int32_t>> idx_to_pos;
      idx_to_pos.reserve(2 * map.ghosts().size());
      for (auto idx : map.ghosts())
      {
        idx_to_pos.push_back(
            {idx, static_cast<std::int32_t>(idx_to_pos.size())});
      }
      std::ranges::sort(idx_to_pos);

      // Build list of (local ghost position, sharing rank) pairs from
      // the received data, and sort
      std::vector<std::pair<std::int32_t, int>> idxpos_to_rank;
      for (std::size_t i = 0; i < recv_indices.size(); i += 2)
      {
        std::int64_t idx = recv_indices[i];
        auto it = std::ranges::lower_bound(
            idx_to_pos, std::pair<std::int64_t, std::int32_t>{idx, 0},
            [](auto a, auto b) { return a.first < b.first; });
        assert(it != idx_to_pos.end() and it->first == idx);

        int rank = recv_indices[i + 1];
        idxpos_to_rank.push_back({it->second, rank});
      }
      std::ranges::sort(idxpos_to_rank);

      // Add processed received data to adjacency list data array, and
      // extend offset array
      std::ranges::transform(idxpos_to_rank, std::back_inserter(data),
                             [](auto x) { return x.second; });
      auto it = idxpos_to_rank.begin();
      for (std::size_t i = 0; i < map.ghosts().size(); ++i)
      {
        auto it1
            = std::find_if(it, idxpos_to_rank.end(), [i](auto x)
                           { return x.first != static_cast<std::int32_t>(i); });
        offsets.push_back(offsets.back() + std::distance(it, it1));
        it = it1;
      }
    }
  }

  // Convert ranks for owned indices from neighbour to global ranks
  std::ranges::transform(idx_to_rank, data.begin(),
                         [&dest](auto x) { return dest[x.second]; });

  return graph::AdjacencyList(std::move(data), std::move(offsets));
}
//-----------------------------------------------------------------------------
