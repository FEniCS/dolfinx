// Copyright (C) 2012 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-07-06
// Last changed: 2012-11-12

#define BOOST_NO_HASH

#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>

#include <dolfin/common/Timer.h>
#include "Graph.h"
#include "BoostGraphOrdering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<int>
  BoostGraphOrdering::compute_cuthill_mckee(const Graph& graph, bool reverse)
{
  Timer timer("Boost Cuthill-McKee graph ordering (from dolfin::Graph)");

  // Number of vertices
  const std::size_t n = graph.size();

  // Typedef for Boost compressed sparse row graph
  typedef boost::compressed_sparse_row_graph<boost::directedS> BoostGraph;

  // Build Boost graph
  const BoostGraph boost_graph = build_csr_directed_graph<BoostGraph>(graph);

  // Check if graph has no edges
  std::vector<int> map(n);
  if (boost::num_edges(boost_graph) == 0)
  {
    // Graph has no edges, so no need to re-order
    for (std::size_t i = 0; i < map.size(); ++i)
      map[i] = i;
  }
  else
  {
    // Boost vertex -> index map
    const boost::property_map<BoostGraph, boost::vertex_index_t>::type
      boost_index_map = get(boost::vertex_index, boost_graph);

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
std::vector<int> BoostGraphOrdering::compute_cuthill_mckee(
  const std::set<std::pair<std::size_t, std::size_t>>& edges,
  std::size_t size, bool reverse)
{
  Timer timer("Boost Cuthill-McKee graph ordering");

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
    for (std::size_t i = 0; i < map.size(); ++i)
      map[i] = i;
  }
  else
  {
    // Get Boost vertex -> index map
    const boost::property_map<BoostGraph, boost::vertex_index_t>::type
      boost_index_map = get(boost::vertex_index, boost_graph);

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
template<typename T, typename X>
T BoostGraphOrdering::build_undirected_graph(const X& graph)
{
  Timer timer("Build Boost undirected graph");

  // Graph size
  const std::size_t n = graph.size();

  // Build Boost graph
  T boost_graph(n);
  typename X::const_iterator vertex;
  graph_set_type::const_iterator edge;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
  {
    const std::size_t vertex_index = vertex - graph.begin();
    for (edge = vertex->begin(); edge != vertex->end(); ++edge)
    {
      if (vertex_index < *edge)
        boost::add_edge(vertex_index, *edge, boost_graph);
    }
  }

  return boost_graph;
}
//-----------------------------------------------------------------------------
template<typename T, typename X>
T BoostGraphOrdering::build_directed_graph(const X& graph)
{
  Timer timer("Build Boost directed graph");

  // Graph size
  const std::size_t n = graph.size();

  // Build Boost graph
  T boost_graph(n);
  typename X::const_iterator vertex;
  graph_set_type::const_iterator edge;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
  {
    const std::size_t vertex_index = vertex - graph.begin();
    for (edge = vertex->begin(); edge != vertex->end(); ++edge)
    {
      if (vertex_index != *edge)
        boost::add_edge(vertex_index, *edge, boost_graph);
    }
  }

  return boost_graph;
}
//-----------------------------------------------------------------------------
template<typename T, typename X>
T BoostGraphOrdering::build_csr_directed_graph(const X& graph)
{
  Timer timer("Build Boost CSR graph");

  // Count number of edges
  Graph::const_iterator vertex;
  std::size_t num_edges = 0;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
    num_edges += vertex->size();

  // Build list of graph edges
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  edges.reserve(num_edges);
  graph_set_type::const_iterator edge;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
    for (edge = vertex->begin(); edge != vertex->end(); ++edge)
      edges.push_back(std::make_pair(vertex - graph.begin(), *edge));

  // Number of vertices
  const std::size_t n = graph.size();

  // Build and return Boost graph
  return T(boost::edges_are_unsorted_multi_pass, edges.begin(), edges.end(), n);
}
//-----------------------------------------------------------------------------
