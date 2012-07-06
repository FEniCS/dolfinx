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

  // Renumber graph (reverse Cuthill--McKee)
  std::vector<uint> inv_perm(boost::num_vertices(boost_graph));
  boost::cuthill_mckee_ordering(boost_graph, inv_perm.begin());

  // Build old-to-new vertex map
  std::vector<dolfin::uint> map(boost::num_vertices(boost_graph));
  for (uint i = 0; i < inv_perm.size(); ++i)
    map[boost_index_map[inv_perm[i]]] = i;

  return map;
}
//-----------------------------------------------------------------------------
