// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-16
// Last changed: 2010-11-25

#ifndef __GRAPH_TYPES_H
#define __GRAPH_TYPES_H

#include <vector>

#define BOOST_NO_HASH

#include <boost/graph/adjacency_list.hpp>
#include <boost/unordered_set.hpp>
#include <dolfin/common/Set.h>

namespace dolfin
{

  /// Typedefs for simple graph data structures

  /// Vector of unordered sets
  //typedef std::vector<boost::unordered_set<unsigned int> > Graph;
  typedef std::vector<dolfin::Set<unsigned int> > Graph;

  // Boost graph typedefs
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> BoostDirectedGraph;
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> BoostUndirectedGraph;
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> BoostBidirectionalGraph;

}

#endif
