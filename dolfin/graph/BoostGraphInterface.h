// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-24
// Last changed:

#ifndef __DOLFIN_BOOST_GRAPH_INTERFACE_H
#define __DOLFIN_BOOST_GRAPH_INTERFACE_H

#include <dolfin/common/Array.h>
#include <dolfin/common/types.h>
#include "Graph.h"

namespace dolfin
{

  class Mesh;

  /// This class colors a graph using the Boost Graph Library.

  class BoostGraphInterface
  {

  public:

    /// Compute vertex colors
    static void compute_local_vertex_coloring(const Graph& graph, Array<uint>& colors);

  };
}

#endif
