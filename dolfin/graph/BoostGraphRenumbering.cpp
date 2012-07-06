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

#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/king_ordering.hpp>
#include <boost/graph/minimum_degree_ordering.hpp>
#include <boost/graph/properties.hpp>

#include "dolfin/log/log.h"
#include "dolfin/common/MPI.h"
#include "Graph.h"
#include "BoostGraphRenumbering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<dolfin::uint> BoostGraphRenumbering::compute_cuthill_mckee(const Graph& graph)
{
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UndirectedGraph;

  // Create Boost graph
  const uint n = graph.size();
  UndirectedGraph boost_graph(n);

  // Build Boost graph
  for (uint i = 0; i < n; ++i)
  {
    for (uint j = 0; j < graph[i].size(); ++j)
    {
      if (i != graph[i][j])
        boost::add_edge(i, graph[i][j], boost_graph);
    }
  }

  boost::property_map<UndirectedGraph, boost::vertex_index_t>::type
    boost_index_map = get(boost::vertex_index, boost_graph);

  // Renumber graph (reverse Cuthill--McKee)
  std::vector<uint> inv_perm(n);
  boost::cuthill_mckee_ordering(boost_graph, inv_perm.begin());

  // Build old-to-new vertex map
  std::vector<dolfin::uint> map(n);
  for (uint i = 0; i < n; ++i)
    map[boost_index_map[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> BoostGraphRenumbering::compute_king(const Graph& graph)
{
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UndirectedGraph;

  // Create Boost graph
  const uint n = graph.size();
  UndirectedGraph boost_graph(n);

  // Build Boost graph
  for (uint i = 0; i < n; ++i)
  {
    for (uint j = 0; j < graph[i].size(); ++j)
    {
      if (i != graph[i][j])
        boost::add_edge(i, graph[i][j], boost_graph);
    }
  }

  boost::property_map<UndirectedGraph, boost::vertex_index_t>::type
    boost_index_map = get(boost::vertex_index, boost_graph);

  // Renumber graph (King)
  std::vector<uint> inv_perm(n);
  boost::king_ordering(boost_graph, inv_perm.rbegin());

  // Build old-to-new vertex map
  std::vector<dolfin::uint> map(n);
  for (uint i = 0; i < n; ++i)
    map[boost_index_map[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint>
  BoostGraphRenumbering::compute_minimum_degree(const Graph& graph, const int delta)
{
  cout << "Start renumbering" << endl;

  typedef double Type;

  using namespace boost;

  //must be BGL directed graph now
  typedef adjacency_list<vecS, vecS, directedS>  BoostGraph;
  typedef graph_traits<BoostGraph>::vertex_descriptor Vertex;

  const uint n = graph.size();

  cout << "n is " << n << endl;

  BoostGraph boost_graph(n);

  // Build Boost graph
  for (uint i = 0; i < n; ++i)
  {
    for (uint j = 0; j < graph[i].size(); ++j)
    {
      if (i != graph[i][j])
        boost::add_edge(i, graph[i][j], boost_graph);
      //if (i < graph[i][j])
      //{
      //  boost::add_edge(i, graph[i][j], boost_graph);
      //  boost::add_edge(graph[i][j], i, boost_graph);
      //}
    }
  }

  /*
  Graph::const_iterator vertex;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
  {
    const uint vertex_index = std::distance(graph.begin(), vertex);
    graph_set_type::const_iterator edge;
    for (edge = vertex->begin(); edge != vertex->end(); ++edge)
    {
      if (vertex_index != *edge)
        boost::add_edge(vertex_index, *edge, boost_graph);
    }
  }
  */

  std::vector<int> inv_perm(n, 0), perm(n, 0), degree(n, 0);
  std::vector<int> supernode_sizes(n, 1);

  boost::property_map<BoostGraph, vertex_index_t>::type
    id = get(vertex_index, boost_graph);

  minimum_degree_ordering
    (boost_graph,
     make_iterator_property_map(&degree[0], id, degree[0]),
     &inv_perm[0],
     &perm[0],
     make_iterator_property_map(&supernode_sizes[0], id, supernode_sizes[0]),
     delta, id);

  cout << "End renumbering" << endl;

  // Build old-to-new vertex map
  std::vector<dolfin::uint> map(n);
  for (uint i = 0; i < n; ++i)
    map[id[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
