// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "AdjacencyList.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dolfinx::graph
{
/// @brief Re-order a graph using the Gibbs-Poole-Stockmeyer algorithm.
///
/// The algorithm is described in *An Algorithm for Reducing the
/// Bandwidth and Profile of a Sparse Matrix*, SIAM Journal on Numerical
/// Analysis, 13(2): 236-250, 1976, https://doi.org/10.1137/0713023.
///
/// The pseudo-diameter search (the dominant cost for dense graphs, e.g.
/// dof-adjacency graphs from higher-order elements) evaluates
/// `num_threads` candidate endpoints concurrently. The result is
/// identical for any `num_threads`, since candidates are only evaluated
/// concurrently, not selected concurrently: the choice of which
/// candidate to use is still made sequentially afterwards, in the
/// original order.
///
/// @param[in] graph The graph to compute a re-ordering for
/// @param[in] num_threads Number of threads to use for the
/// pseudo-diameter candidate search. `1` runs serially.
/// @return Reordering array `map`, where `map[i]` is the new index of
/// node `i`.
std::vector<std::int32_t>
reorder_gps(const graph::AdjacencyList<std::int32_t>& graph,
            std::size_t num_threads);

} // namespace dolfinx::graph
