// Copyright (C) 2012 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Graph.h"
#include <set>
#include <utility>
#include <vector>

namespace dolfin
{

/// This class computes graph re-orderings. It uses Boost Graph.

class BoostGraphOrdering
{

public:
  /// Compute re-ordering (map[old] -> new) using Cuthill-McKee
  /// algorithm
  static std::vector<int> compute_cuthill_mckee(const Graph& graph,
                                                bool reverse = false);

  /// Compute re-ordering (map[old] -> new) using Cuthill-McKee
  /// algorithm
  static std::vector<int> compute_cuthill_mckee(
      const std::set<std::pair<std::size_t, std::size_t>>& edges,
      std::size_t size, bool reverse = false);

private:
  // Build Boost undirected graph
  template <typename T, typename X>
  static T build_undirected_graph(const X& graph);

  // Build Boost directed graph
  template <typename T, typename X>
  static T build_directed_graph(const X& graph);

  // Build Boost compressed sparse row graph
  template <typename T, typename X>
  static T build_csr_directed_graph(const X& graph);
};
}


