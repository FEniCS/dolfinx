// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "AdjacencyList.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mpi.h>
#include <span>
#include <variant>
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

/// @brief Signature of functions that reorder the nodes of a graph.
using reorder_fn = std::function<std::vector<std::int32_t>(
    const graph::AdjacencyList<std::int32_t>&)>;

/// @brief Signature of functions that reorder points from their positions.
///
/// @param[in] comm Communicator over which the points are distributed.
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point.
/// @return Reordering array `map`, where `map[i]` is the new index of point
/// `i`.
using geom_reorder_fn = std::function<std::vector<std::int32_t>(
    MPI_Comm comm, std::span<const double> x, int gdim)>;

/// @brief A graph or geometric reordering function.
using AnyReorderFunction = std::variant<reorder_fn, geom_reorder_fn>;

/// @brief A reordering function for mesh cells.
///
/// The default reordering is ::reorder_rcm. A ::geom_reorder_fn is called with
/// the centroids of the locally owned cells.
struct Reorder
{
  /// Reordering function. Defaults to ::reorder_rcm.
  AnyReorderFunction fn = reorder_fn(reorder_rcm);
};

} // namespace dolfinx::graph
