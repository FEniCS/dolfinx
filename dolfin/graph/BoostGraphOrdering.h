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
// Last changed: 2012-11-12

#ifndef __DOLFIN_BOOST_GRAPH_ORDERING_H
#define __DOLFIN_BOOST_GRAPH_ORDERING_H

#include <set>
#include <utility>
#include <vector>
#include "Graph.h"

namespace dolfin
{

  /// This class computes graph re-orderings. It uses Boost Graph.

  class BoostGraphOrdering
  {

  public:

    /// Compute re-ordering (map[old] -> new) using Cuthill-McKee algorithm
    static std::vector<int> compute_cuthill_mckee(const Graph& graph,
                                                  bool reverse=false);

    /// Compute re-ordering (map[old] -> new) using Cuthill-McKee algorithm
    static std::vector<int>
      compute_cuthill_mckee(const std::set<std::pair<std::size_t, std::size_t>>& edges,
                            std::size_t size, bool reverse=false);

  private:

    // Build Boost undirected graph
    template<typename T, typename X>
    static T build_undirected_graph(const X& graph);

    // Build Boost directed graph
    template<typename T, typename X>
    static T build_directed_graph(const X& graph);

    // Build Boost compressed sparse row graph
    template<typename T, typename X>
    static T build_csr_directed_graph(const X& graph);

  };

}

#endif
