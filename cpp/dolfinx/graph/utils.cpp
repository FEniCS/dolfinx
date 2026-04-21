// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <algorithm>
#include <dolfinx/common/MPI.h>
#include <format>
#include <sstream>
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

  std::stringstream out;
  out << std::format("{{\"directed\": true, \"multigraph\": false, \"graph\": "
                     "[], \"nodes\": [");
  for (std::int32_t n = 0; n < g.num_nodes(); ++n)
  {
    // Note: it is helpful to order map keys alphabetically
    out << std::format("{{\"num_ghosts\": {}, \"weight\": {},  \"id\": {}}}",
                       node_weights[n].second, node_weights[n].first, n);
    if (n != g.num_nodes() - 1)
      out << ", ";
  }
  out << "], ";
  out << "\"adjacency\": [";
  for (std::int32_t n = 0; n < g.num_nodes(); ++n)
  {
    out << "[";
    auto links = g.links(n);
    for (std::size_t edge = 0; edge < links.size(); ++edge)
    {
      auto [e, w, local] = links[edge];
      out << std::format("{{\"local\": {}, \"weight\": {}, \"id\": {}}}", local,
                         w, e);
      if (edge != links.size() - 1)
        out << ", ";
    }
    out << "]";
    if (n != g.num_nodes() - 1)
      out << ", ";
  }
  out << "]}";

  return out.str();
}
//-----------------------------------------------------------------------------
