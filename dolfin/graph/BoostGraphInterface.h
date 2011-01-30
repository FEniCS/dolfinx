// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-24
// Last changed:

#ifndef __DOLFIN_BOOST_GRAPH_INTERFACE_H
#define __DOLFIN_BOOST_GRAPH_INTERFACE_H

#include <boost/graph/sequential_vertex_coloring.hpp>
#include <dolfin/common/Array.h>
#include <dolfin/common/types.h>
#include <dolfin/graph/Graph.h>

namespace dolfin
{

  template<class T> class Array;
  class Mesh;

  /// This class colors a graph using the Boost Graph Library.

  class BoostGraphInterface
  {

  public:

    /// Compute vertex colors
    static void compute_local_vertex_coloring(const Graph& graph, Array<uint>& colors);

    /// Compute vertex colors
    template<class T>
    static uint compute_local_vertex_coloring(const T& graph, Array<uint>& colors)
    {
      // Number of vertices in graph
      const uint num_vertices = boost::num_vertices(graph);
      assert(num_vertices == colors.size());

      typedef typename boost::graph_traits<T>::vertex_descriptor vert_descriptor;
      typedef typename boost::graph_traits<T>::vertex_iterator vert_iterator;
      typedef typename boost::graph_traits<T>::vertices_size_type vert_size_type;
      typedef typename boost::property_map<T, boost::vertex_index_t>::const_type vert_index_map;

      // Create vector to hold colors
      std::vector<vert_size_type> color_vec(num_vertices);

      // Color vertices
      std::cout << "Start Boost coloring." <<  std::endl;
      boost::iterator_property_map<vert_size_type*, vert_index_map> color(&color_vec.front(), get(boost::vertex_index, graph));
      const vert_size_type num_colors = sequential_vertex_coloring(graph, color);
      std::cout << "Boost coloring finished." <<  std::endl;

      // Copy result into Array
      assert(colors.size() == color_vec.size());
      for (uint i = 0; i < num_vertices; ++i)
        colors[i] = color_vec[i];

      std::cout << "Number of colors: " << num_colors << std::endl;
      return num_colors;
    }

  };
}

#endif
