// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-07-06
// Last changed:

#ifndef __DOLFIN_BOOST_GRAPH_RENUMBERING_H
#define __DOLFIN_BOOST_GRAPH_RENUMBERING_H

#include <vector>
#include "Graph.h"

namespace dolfin
{

  /// This class computes graph re-orderings. It uses Boost Graph.

  class BoostGraphRenumbering
  {

  public:

    /// Compute renumbering (map[old] -> new) using Cuthill-McKee algorithm
    static std::vector<uint> compute_cuthill_mckee(const Graph& graph, bool reverse=false);

    /// Compute renumbering (map[old] -> new) using King algorithm
    static std::vector<uint> compute_king(const Graph& graph);

    /// Compute renumbering (map[old] -> new) using minimum degree algorithm
    static std::vector<uint> compute_minimum_degree(const Graph& graph,
                                                    const int delta=0);

  private:

    // Build Boost undirected graph
    template<typename T>
    static T build_undirected_graph(const Graph& graph);

    // Build Boost directed graph
    template<typename T>
    static T build_directed_graph(const Graph& graph);

  };

}

#endif
