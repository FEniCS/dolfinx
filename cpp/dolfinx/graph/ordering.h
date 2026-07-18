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
/// At each step the pseudo-diameter search (the dominant cost for
/// dense graphs, e.g. dof-adjacency graphs from higher-order elements)
/// tests candidate root vertices from the final level of a level
/// structure, in increasing degree order, to find the pair of vertices
/// that minimises level width. Testing every candidate costs
/// O(|S| * (V+E)) per step, where |S| is the size of the final level,
/// which can dominate run time for large graphs. `max_candidates`
/// bounds how many of the lowest-degree candidates are tried at each
/// step; the (up to `max_candidates`) candidates that are tried are
/// evaluated concurrently across `num_threads` threads. The result is
/// identical for any `num_threads`, since candidates are only
/// evaluated concurrently, not selected concurrently: the choice of
/// which candidate to use is still made sequentially afterwards, in
/// the original order.
///
/// A small `max_candidates` (e.g. 5) is a good trade-off for simple,
/// convex, roughly uniform-density meshes, where it matches an
/// exhaustive search at a fraction of the cost. For non-convex domains
/// or strongly graded meshes (e.g. boundary-layer meshes) a small cap
/// can measurably worsen the resulting bandwidth and profile; pass a
/// larger value if ordering quality matters more than reordering time
/// for such meshes.
///
/// To recover the original (uncapped) Gibbs-Poole-Stockmeyer algorithm
/// as published -- testing every candidate in the final level, with no
/// truncation -- pass `max_candidates =
/// std::numeric_limits<std::size_t>::max()` (or any value at least as
/// large as the largest final level `S` encountered; the actual number
/// of candidates tested is `std::min(max_candidates, S.size())`, so an
/// oversized value is never invalid, just unnecessary).
///
/// @param[in] graph The graph to compute a re-ordering for
/// @param[in] max_candidates Maximum number of final-level candidates
/// to test at each step of the pseudo-diameter search. Pass
/// `std::numeric_limits<std::size_t>::max()` for the original,
/// exhaustive algorithm (see above).
/// @param[in] num_threads Number of threads to use for the
/// pseudo-diameter candidate search. `1` runs serially.
/// @return Reordering array `map`, where `map[i]` is the new index of
/// node `i`.
std::vector<std::int32_t>
reorder_gps(const graph::AdjacencyList<std::int32_t>& graph,
            std::size_t max_candidates, std::size_t num_threads);

/// @brief Re-order a graph using the Reverse Cuthill-McKee algorithm.
///
/// The algorithm is described in *Reducing the Bandwidth of Sparse
/// Symmetric Matrices*, Proceedings of the 1969 24th National
/// Conference, ACM, 1969, pp. 157-172,
/// https://doi.org/10.1145/800195.805928. The pseudo-peripheral root
/// used to start the ordering is found using the George-Liu "double
/// sweep" heuristic (as in `reorder_gps`'s Algorithm I, but trying only
/// the single lowest-degree candidate at each step).
///
/// Unlike `reorder_gps`, there is no width-minimising second phase: a
/// single level structure is built from the pseudo-peripheral root,
/// each level is numbered in increasing degree order, and the whole
/// numbering is reversed (the "reverse" in Reverse Cuthill-McKee, which
/// tends to reduce profile relative to the plain, non-reversed
/// numbering). This makes `reorder_rcm` an O(V+E) algorithm with a much
/// smaller constant than `reorder_gps`, at the cost of the bandwidth
/// and profile quality that `reorder_gps`'s extra phase can provide on
/// non-convex or strongly graded meshes -- for simple, convex,
/// roughly uniform-density meshes the two typically produce comparable
/// bandwidth.
///
/// @param[in] graph The graph to compute a re-ordering for
/// @return Reordering array `map`, where `map[i]` is the new index of
/// node `i`.
std::vector<std::int32_t>
reorder_rcm(const graph::AdjacencyList<std::int32_t>& graph);

} // namespace dolfinx::graph
