// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-03-19
// Last changed: 2007-03-19

#ifndef __UNDIRECTED_CLIQUE_H
#define  __UNDIRECTED_CLIQUE_H

#include <dolfin/Graph.h>

namespace dolfin
{
  /// A undirected graph where all vertices are adjacent to each other.
  /// The number of vertices is given by num_vertices >= 1. The number of 
  /// edges is given by ((num_vertices - 1) * num_vertices) / 2

  class UndirectedClique : public Graph
  {
  public:

    UndirectedClique(uint num_vertices);

  };
  
}

#endif
