// Copyright (C) 2011 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-21
// Last changed:

#ifndef __GRAPH_COLORING_H
#define __GRAPH_COLORING_H

#include <dolfin/common/types.h>
#include "Graph.h"

namespace dolfin
{

  template<class T> class Array;
  class Mesh;

  /// This class provides a common interface to graph coloring libraries

  class GraphColoring
  {

  public:

    /// Compute vertex colors
    static uint compute_local_vertex_coloring(const Graph& graph,
                                              Array<uint>& colors);

  };
}

#endif
