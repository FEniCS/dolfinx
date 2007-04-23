// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-01
// Last changed: 2007-01-01

#ifndef __ADJACENCY_GRAPH_H
#define __ADJACENCY_GRAPH_H

#include <boost/graph/adjacency_list.hpp>

namespace dolfin
{

  /// AdjacencyGraph is a typedef for an undirected boost::adjacency_list
  /// with the outer container (vertices) chosen as an std::vector and
  /// the inner container (edges of vertices) chosen as an std::set.

  typedef boost::adjacency_list<boost::setS,
                                boost::vecS,
                                boost::undirectedS,
                                boost::no_property,
                                boost::no_property> AdjacencyGraph;

}

#endif
