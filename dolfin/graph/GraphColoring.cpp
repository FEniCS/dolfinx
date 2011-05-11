// Copyright (C) 2011 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
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
