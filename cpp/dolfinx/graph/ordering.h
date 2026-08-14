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
/// @brief Re-order a graph using the Reverse Cuthill-McKee algorithm.
///
/// The algorithm is described in *Reducing the Bandwidth of Sparse
/// Symmetric Matrices*, Proceedings of the 1969 24th National
/// Conference, ACM, 1969, pp. 157-172,
/// https://doi.org/10.1145/800195.805928. The pseudo-peripheral root
/// used to start the ordering is found using the George-Liu "double
/// sweep" heuristic, trying only the single lowest-degree candidate at
/// each step.
///
/// A single level structure is built from the pseudo-peripheral root,
/// each level is numbered in increasing degree order, and the whole
/// numbering is reversed (the "reverse" in Reverse Cuthill-McKee, which
/// tends to reduce profile relative to the plain, non-reversed
/// numbering). This makes `reorder_rcm` an O(V+E) algorithm with a
/// small constant.
///
/// @param[in] graph The graph to compute a re-ordering for
/// @return Reordering array `map`, where `map[i]` is the new index of
/// node `i`.
std::vector<std::int32_t>
reorder_rcm(const graph::AdjacencyList<std::int32_t>& graph);

} // namespace dolfinx::graph
