// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <format>
#include <sstream>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::string common::comm_to_json(
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
