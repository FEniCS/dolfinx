// Copyright (C) 2010 Garth N. Wells
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
// First added:  2010-11-24
// Last changed:

#include <iostream>
#include <vector>

#define BOOST_NO_HASH

#include <boost/foreach.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>

#include <dolfin/log/log.h>
#include "Graph.h"
#include "BoostGraphColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::size_t BoostGraphColoring::compute_local_vertex_coloring(const Graph& graph,
                                                 std::vector<std::size_t>& colors)
{
  // Number of vertices
  const std::size_t n = graph.size();
  dolfin_assert(n == colors.size());

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
  BoostGraph g(boost::edges_are_unsorted_multi_pass,
               edges.begin(), edges.end(), n);

  // Perform coloring
  return compute_local_vertex_coloring(g, colors);
}
//----------------------------------------------------------------------------
