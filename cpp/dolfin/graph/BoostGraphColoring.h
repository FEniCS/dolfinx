// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Graph.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/sequential_vertex_coloring.hpp>
#include <dolfin/common/Timer.h>
#include <iostream>
#include <vector>

namespace dolfin
{

class Mesh;

namespace graph
{

/// This class colors a graph using the Boost Graph Library.

class BoostGraphColoring
{

public:
  /// Compute vertex colors
  template <typename ColorType>
  static std::size_t
  compute_local_vertex_coloring(const Graph& graph,
                                std::vector<ColorType>& colors)
  {
    Timer timer("Boost graph coloring (from dolfin::Graph)");

    // Typedef for Boost compressed sparse row graph
    typedef boost::
        compressed_sparse_row_graph<boost::directedS,
                                    boost::property<boost::vertex_color_t,
                                                    ColorType>>
            BoostGraph;

    // Number of vertices
    const std::size_t n = graph.size();

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
    {
      for (edge = vertex->begin(); edge != vertex->end(); ++edge)
      {
        const std::size_t vertex_index = vertex - graph.begin();
        if (vertex_index != (std::size_t)*edge)
          edges.push_back(std::make_pair(vertex_index, *edge));
      }
    }
    // Build Boost graph
    const BoostGraph g(boost::edges_are_unsorted_multi_pass, edges.begin(),
                       edges.end(), n);

    // Resize vector to hold colors
    colors.resize(n);

    // Perform coloring
    return compute_local_vertex_coloring(g, colors);
  }

  /// Compute vertex colors
  template <typename T, typename ColorType>
  static std::size_t
  compute_local_vertex_coloring(const T& graph, std::vector<ColorType>& colors)
  {
    Timer timer("Boost graph coloring");

    // Number of vertices in graph
    const std::size_t num_vertices = boost::num_vertices(graph);
    assert(num_vertices == colors.size());

    typedef typename boost::graph_traits<T>::vertices_size_type vert_size_type;
    typedef typename boost::property_map<T, boost::vertex_index_t>::const_type
        vert_index_map;

    // Resize to hold colors
    colors.resize(num_vertices);

    // Color vertices
    std::vector<vert_size_type> _colors(num_vertices);
    boost::iterator_property_map<vert_size_type*, vert_index_map> color(
        &_colors.front(), get(boost::vertex_index, graph));
    const vert_size_type num_colors = sequential_vertex_coloring(graph, color);

    // Copy colors and return
    std::copy(_colors.begin(), _colors.end(), colors.begin());
    return num_colors;
  }
};
}
}