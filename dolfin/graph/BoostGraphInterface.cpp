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
#include <boost/graph/sequential_vertex_coloring.hpp>

#include "dolfin/log/log.h"
#include "dolfin/common/Timer.h"
#include "BoostGraphInterface.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void BoostGraphInterface::compute_local_vertex_coloring(const Graph& graph,
                                                        Array<uint>& colors)
{
  // Number of vertices in graph
  const uint num_vertices = graph.size();
  assert(num_vertices == colors.size());

  //typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS> BoostGraph;
  //typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> BoostGraph;
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> BoostGraph;

  typedef boost::graph_traits<BoostGraph>::vertex_descriptor vert_descriptor;
  typedef boost::graph_traits<BoostGraph>::vertex_iterator vert_iterator;
  typedef boost::graph_traits<BoostGraph>::vertices_size_type vert_size_type;
  typedef boost::property_map<BoostGraph, boost::vertex_index_t>::const_type vert_index_map;

  // Copy Graph data structure into a BoostGraph
  BoostGraph g(num_vertices);
  std::cout << "Building Boost graph." <<  std::endl;
  for (uint i = 0; i < graph.size(); ++i)
  {
    BOOST_FOREACH(boost::unordered_set<uint>::value_type edge, graph[i])
    {
      const uint e = edge;
      boost::add_edge(i, e, g);
    }
  }
  std::cout << "Finished building Boost graph." <<  std::endl;

  // Create vector to hold colors
  std::vector<vert_size_type> color_vec(num_vertices);

  // Color vertices
  std::cout << "Start Boost coloring." <<  std::endl;
  boost::iterator_property_map<vert_size_type*, vert_index_map> color(&color_vec.front(), get(boost::vertex_index, g));
  const vert_size_type num_colors = sequential_vertex_coloring(g, color);
  std::cout << "Boost coloring finished." <<  std::endl;

  // Copy result into Array
  assert(colors.size() == color_vec.size());
  for (uint i = 0; i < num_vertices; ++i)
    colors[i] = color_vec[i];

  std::cout << "Number of colors: " << num_colors << std::endl;
}
//-----------------------------------------------------------------------------
