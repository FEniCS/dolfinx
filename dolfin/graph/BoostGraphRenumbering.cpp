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
std::vector<dolfin::uint> BoostGraphRenumbering::compute_local_renumbering_map(const Graph& graph)
{
  // Create Boost graph
  const uint num_verticies = graph.size();
  BoostUndirectedGraph boost_graph(num_verticies);

  // Build Boost graph
  for (uint i = 0; i < num_verticies; ++i)
  {
    for (uint j = 0; j < graph[i].size(); ++j)
      boost::add_edge(i, graph[i][j], boost_graph);
  }

  boost::property_map<BoostUndirectedGraph, boost::vertex_index_t>::type
    boost_index_map = get(boost::vertex_index, boost_graph);

  //for (uint i = 0; i < graph.size(); ++i)
  //  cout << i << ": " << index_map[i] << endl;


  std::vector<uint> inv_perm(boost::num_vertices(boost_graph));

  // Renumber graph (reverse Cuthill--McKee)
  //boost::cuthill_mckee_ordering(boost_graph, inv_perm.begin());

  // Renumber graph (King)
  //boost::king_ordering(boost_graph, inv_perm.rbegin());


  cout << "Start renumbering" << endl;
  // Renumber graph (minimum degree)
  std::vector<uint> perm(boost::num_vertices(boost_graph));
  std::vector<uint> degree(boost::num_vertices(boost_graph), 0);
  std::vector<uint> super_node_sizes(boost::num_vertices(boost_graph), 1);

  int delta = -1;

  boost::minimum_degree_ordering(boost_graph,
     boost::make_iterator_property_map(&degree[0], boost_index_map, degree[0]),
     &inv_perm[0],
     &perm[0],
     boost::make_iterator_property_map(&super_node_sizes[0], boost_index_map, super_node_sizes[0]),
     delta, boost_index_map);

  cout << "End renumbering" << endl;

  // Build old-to-new vertex map
  std::vector<dolfin::uint> map(boost::num_vertices(boost_graph));
  for (uint i = 0; i < inv_perm.size(); ++i)
    map[boost_index_map[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
