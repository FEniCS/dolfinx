// Copyright (C) 2011 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-21
// Last changed:

#include <string>
#include <dolfin/common/Array.h>
#include <dolfin/log/log.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "BoostGraphInterface.h"
#include "Graph.h"
#include "ZoltanInterface.h"
#include "GraphColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint GraphColoring::compute_local_vertex_coloring(const Graph& graph,
                                                        Array<uint>& colors)
{
  // Get coloring library from parameter system
  const std::string colorer = parameters["graph_coloring_library"];

  // Color mesh
  if (colorer == "Boost")
    return BoostGraphInterface::compute_local_vertex_coloring(graph, colors);
  else if (colorer == "Zoltan")
    return ZoltanInterface::compute_local_vertex_coloring(graph, colors);
  else
  {
    error("Mesh colorer type unkown. Possible options are \"Boost\" or \"Zoltan\".");
    return 0;
  }
}
//----------------------------------------------------------------------------
