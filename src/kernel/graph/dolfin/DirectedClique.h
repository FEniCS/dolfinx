// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-03-19
// Last changed: 2007-03-19

#ifndef __DIRECTED_CLIQUE_H
#define __DIRECTED_CLIQUE_H

#include <dolfin/Graph.h>

namespace dolfin
{
  /// A directed graph where all vertices are adjacent to each other.
  /// The number of vertices is given by num_vertices >= 1. The number of 
  /// edges is given by (num_vertices - 1) * num_vertices)

  class DirectedClique : public Graph
  {
  public:

    DirectedClique(uint num_vertices);

  };
  
}

#endif
