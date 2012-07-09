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

#include <boost/foreach.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <dolfin/common/Array.h>
#include "dolfin/common/Timer.h"
#include "dolfin/log/log.h"
#include "Graph.h"
#include "BoostGraphInterface.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint BoostGraphInterface::compute_local_vertex_coloring(const Graph& graph,
                                                        Array<uint>& colors)
{
  // Number of vertices in graph
  const uint num_vertices = graph.size();
  dolfin_assert(num_vertices == colors.size());

  // Copy Graph data structure into a BoostGraph
  BoostBidirectionalGraph g(num_vertices);
  for (uint i = 0; i < graph.size(); ++i)
  {
    BOOST_FOREACH(boost::unordered_set<uint>::value_type edge, graph[i])
    {
      const uint e = edge;
      boost::add_edge(i, e, g);
    }
  }

  // Perform coloring
  return compute_local_vertex_coloring(g, colors);
}
//----------------------------------------------------------------------------
