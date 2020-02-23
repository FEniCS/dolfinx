// Copyright (C) 2012 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <set>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace graph
{

template <typename T>
class AdjacencyList;

/// This class computes graph re-orderings. It uses Boost Graph.

class BoostGraphOrdering
{

public:
  /// Compute re-ordering (map[old] -> new) using Cuthill-McKee
  /// algorithm
  static std::vector<int>
  compute_cuthill_mckee(const AdjacencyList<std::int32_t>& graph,
                        bool reverse = false);

  /// Compute re-ordering (map[old] -> new) using Cuthill-McKee
  /// algorithm
  static std::vector<int> compute_cuthill_mckee(
      const std::set<std::pair<std::size_t, std::size_t>>& edges,
      std::size_t size, bool reverse = false);
};
} // namespace graph
} // namespace dolfinx
