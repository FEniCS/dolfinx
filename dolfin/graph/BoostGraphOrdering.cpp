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
// Last changed:

#define BOOST_NO_HASH

#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/king_ordering.hpp>
#include <boost/graph/minimum_degree_ordering.hpp>
#include <boost/graph/properties.hpp>

#include "dolfin/common/types.h"
#include "Graph.h"
#include "BoostGraphOrdering.h"

#include "dolfin/log/LogStream.h"
#include "dolfin/common/timing.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<std::size_t>
  BoostGraphOrdering::compute_cuthill_mckee(const Graph& graph, bool reverse)
{
  // Number of vertices
  const std::size_t n = graph.size();

  // Typedef for Boost compressed sparse row graph
  typedef boost::compressed_sparse_row_graph<boost::directedS> BoostGraph;

  // Count number of edges
  Graph::const_iterator vertex;
  std::size_t num_edges = 0;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
    num_edges += vertex->size();

  // Build list of graph edges
  std::vector<std::pair<std::size_t, std::size_t> > edges;
  edges.reserve(num_edges);
  graph_set_type::const_iterator edge;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
    for (edge = vertex->begin(); edge != vertex->end(); ++edge)
      edges.push_back(std::make_pair(vertex - graph.begin(), *edge));

  // Build Boost graph
  BoostGraph boost_graph(boost::edges_are_unsorted_multi_pass,
                         edges.begin(), edges.end(), n);

  // Boost vertex -> index map
  boost::property_map<BoostGraph, boost::vertex_index_t>::type
    boost_index_map = get(boost::vertex_index, boost_graph);

  // Compute graph re-ordering
  std::vector<std::size_t> inv_perm(n);
  if (!reverse)
    boost::cuthill_mckee_ordering(boost_graph, inv_perm.begin());
  else
    boost::cuthill_mckee_ordering(boost_graph, inv_perm.rbegin());

  // Build old-to-new vertex map
  std::vector<std::size_t> map(n);
  for (std::size_t i = 0; i < n; ++i)
    map[boost_index_map[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> BoostGraphOrdering::compute_king(const Graph& graph)
{
  // Typedef for Boost undirected graph
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UndirectedGraph;

  // Graph size
  const uint n = graph.size();

  // Build Boost graph
  UndirectedGraph boost_graph = build_undirected_graph<UndirectedGraph>(graph);

  // Compute graph re-ordering
  std::vector<uint> inv_perm(n);
  boost::king_ordering(boost_graph, inv_perm.rbegin());

  // Boost vertex -> index map
  boost::property_map<UndirectedGraph, boost::vertex_index_t>::type
    boost_index_map = get(boost::vertex_index, boost_graph);

  // Build old-to-new vertex map
  std::vector<dolfin::uint> map(n);
  for (uint i = 0; i < n; ++i)
    map[boost_index_map[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> BoostGraphOrdering::compute_king(const std::vector<std::vector<uint> >& graph)
{
  // Typedef for Boost undirected graph
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UndirectedGraph;

  // Graph size
  const uint n = graph.size();

  // Build Boost graph
  UndirectedGraph boost_graph = build_undirected_graph<UndirectedGraph>(graph);

  // Compute graph re-ordering
  std::vector<uint> inv_perm(n);
  boost::king_ordering(boost_graph, inv_perm.rbegin());

  // Boost vertex -> index map
  boost::property_map<UndirectedGraph, boost::vertex_index_t>::type
    boost_index_map = get(boost::vertex_index, boost_graph);

  // Build old-to-new vertex map
  std::vector<dolfin::uint> map(n);
  for (uint i = 0; i < n; ++i)
    map[boost_index_map[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint>
  BoostGraphOrdering::compute_minimum_degree(const Graph& graph, const int delta)
{
  // Typedef for Boost directed graph
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> BoostGraph;

  // Graph size
  const uint n = graph.size();

  // Build Boost directed graph
  BoostGraph boost_graph = build_directed_graph<BoostGraph>(graph);

  // Boost vertex -> index map
  boost::property_map<BoostGraph, boost::vertex_index_t>::type
    boost_index_map = get(boost::vertex_index, boost_graph);

  // Compute graph re-ordering
  std::vector<int> inv_perm(n, 0), perm(n, 0), degree(n, 0);
  std::vector<int> supernode_sizes(n, 1);
  boost::minimum_degree_ordering(boost_graph,
     make_iterator_property_map(degree.begin(), boost_index_map, degree[0]),
     inv_perm.begin(), perm.begin(),
     make_iterator_property_map(supernode_sizes.begin(), boost_index_map, supernode_sizes[0]),
     delta, boost_index_map);

  // Build old-to-new vertex map
  std::vector<dolfin::uint> map(n);
  for (uint i = 0; i < n; ++i)
    map[boost_index_map[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
template<typename T, typename X>
T BoostGraphOrdering::build_undirected_graph(const X& graph)
{
  // Graph size
  const uint n = graph.size();

  // Build Boost graph
  T boost_graph(n);
  typename X::const_iterator vertex;
  graph_set_type::const_iterator edge;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
  {
    const uint vertex_index = vertex - graph.begin();
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
  // Graph size
  const uint n = graph.size();

  // Build Boost graph
  T boost_graph(n);
  typename X::const_iterator vertex;
  graph_set_type::const_iterator edge;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
  {
    const uint vertex_index = vertex - graph.begin();
    for (edge = vertex->begin(); edge != vertex->end(); ++edge)
    {
      if (vertex_index != *edge)
        boost::add_edge(vertex_index, *edge, boost_graph);
    }
  }

  return boost_graph;
}
//-----------------------------------------------------------------------------
