// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-24
// Last changed:

#include <iostream>
#include <vector>

// This is to avoid a GCC 4.3+ error
// FIXME: Check that it does not impact on performance
#define BOOST_NO_HASH

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
  assert(num_vertices == colors.size());

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
