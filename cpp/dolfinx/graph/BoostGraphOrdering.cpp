// Copyright (C) 2012 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#define BOOST_NO_HASH

#include "BoostGraphOrdering.h"
#include "AdjacencyList.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <dolfinx/common/Timer.h>
#include <numeric>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
template <typename T, typename X>
T build_csr_directed_graph(const graph::AdjacencyList<X>& graph)
{
  common::Timer timer("Build Boost CSR graph");

  // Build list of graph edges
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  edges.reserve(graph.array().rows());
  for (int v = 0; v < graph.num_nodes(); ++v)
  {
    auto links = graph.links(v);
    for (int e = 0; e < links.rows(); ++e)
      edges.push_back({v, links[e]});
  }

  // Number of vertices
  const std::size_t n = graph.num_nodes();

  // Build and return Boost graph
  return T(boost::edges_are_unsorted_multi_pass, edges.begin(), edges.end(), n);
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
std::vector<int> dolfinx::graph::BoostGraphOrdering::compute_cuthill_mckee(
    const graph::AdjacencyList<std::int32_t>& graph, bool reverse)
{
  common::Timer timer(
      "Boost Cuthill-McKee graph ordering (from dolfinx::Graph)");

  // Number of vertices
  const std::size_t n = graph.num_nodes();

  // Typedef for Boost compressed sparse row graph
  typedef boost::compressed_sparse_row_graph<boost::directedS> BoostGraph;

  // Build Boost graph
  const BoostGraph boost_graph = build_csr_directed_graph<BoostGraph>(graph);

  // Check if graph has no edges
  std::vector<int> map(n);
  if (boost::num_edges(boost_graph) == 0)
  {
    // Graph has no edges, so no need to re-order
    std::iota(map.begin(), map.end(), 0);
  }
  else
  {
    // Boost vertex -> index map
    const boost::property_map<BoostGraph, boost::vertex_index_t>::type
        boost_index_map
        = get(boost::vertex_index, boost_graph);

    // Compute graph re-ordering
    std::vector<int> inv_perm(n);
    if (!reverse)
      boost::cuthill_mckee_ordering(boost_graph, inv_perm.begin());
    else
      boost::cuthill_mckee_ordering(boost_graph, inv_perm.rbegin());

    // Build old-to-new vertex map
    for (std::size_t i = 0; i < map.size(); ++i)
      map[boost_index_map[inv_perm[i]]] = i;
  }

  return map;
}
//-----------------------------------------------------------------------------
std::vector<int> dolfinx::graph::BoostGraphOrdering::compute_cuthill_mckee(
    const std::set<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t size, bool reverse)
{
  common::Timer timer("Boost Cuthill-McKee graph ordering");

  // Typedef for Boost compressed sparse row graph
  typedef boost::compressed_sparse_row_graph<boost::directedS> BoostGraph;

  // Build Boost graph
  const BoostGraph boost_graph(boost::edges_are_unsorted_multi_pass,
                               edges.begin(), edges.end(), size);

  // Check if graph has no edges
  std::vector<int> map(size);
  if (boost::num_edges(boost_graph) == 0)
  {
    // Graph has no edges, so no need to re-order
    std::iota(map.begin(), map.end(), 0);
  }
  else
  {
    // Get Boost vertex -> index map
    const boost::property_map<BoostGraph, boost::vertex_index_t>::type
        boost_index_map
        = get(boost::vertex_index, boost_graph);

    // Compute graph re-ordering
    std::vector<int> inv_perm(size);
    if (!reverse)
      boost::cuthill_mckee_ordering(boost_graph, inv_perm.begin());
    else
      boost::cuthill_mckee_ordering(boost_graph, inv_perm.rbegin());

    // Build old-to-new vertex map
    for (std::size_t i = 0; i < size; ++i)
      map[boost_index_map[inv_perm[i]]] = i;
  }

  return map;
}
//-----------------------------------------------------------------------------
