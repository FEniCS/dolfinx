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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-02-21
// Last changed:

#ifndef __GRAPH_COLORING_H
#define __GRAPH_COLORING_H


#include <cstddef>
#include <vector>
#include "Graph.h"

namespace dolfin
{

  /// This class provides a common interface to graph coloring libraries

  class GraphColoring
  {

  public:

    /// Compute vertex colors
    static std::size_t
      compute_local_vertex_coloring(const Graph& graph,
                                    std::vector<std::size_t>& colors);

  };
}

#endif
