// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <vector>

namespace dolfinx::graph
{
template <typename T>
class AdjacencyList;

/// @brief Re-order a graph using the Gibbs-Poole-Stockmeyer algorithm.
///
/// The algorithm is described in *An Algorithm for Reducing the
/// Bandwidth and Profile of a Sparse Matrix*, SIAM Journal on Numerical
/// Analysis, 13(2): 236-250, 1976, https://doi.org/10.1137/0713023.
///
/// @param[in] graph The graph to compute a re-ordering for
/// @return Reordering array `map`, where `map[i]` is the new index of
/// node `i`
std::vector<std::int32_t>
reorder_gps(const graph::AdjacencyList<std::int32_t>& graph);

} // namespace dolfinx::graph