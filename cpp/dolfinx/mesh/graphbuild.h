// Copyright (C) 2010-2026 Garth N. Wells, Paul T. Kühner and Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <mpi.h>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace dolfinx::mesh
{
enum class CellType : std::int8_t;

/// @brief Build the local dual graph with optional cell-facet weights.
///
/// @param[in] celltypes Cell types, as in the unweighted overload.
/// @param[in] cells Flattened cell vertices, one array per cell type.
/// @param[in] max_facet_to_cell_links See the unweighted overload.
/// @param[in] num_threads Number of threads; must be positive.
/// @param[in] facet_weights One array per cell type, flattened in cell-major
/// order using the local facet numbering from get_entity_vertices. Each array
/// has num_cells * num_facets entries. Weights must be positive integers.
/// An empty outer span selects the unweighted path, which allocates no weight
/// buffers. All weights on a shared facet are averaged (rounded down); this
/// value is assigned to both directions of every cell-pair edge on that facet.
/// @return A tuple of six items:
/// 1. The local dual graph, with edges sorted by vertex key.
/// 2. The unmatched facets, flattened in cell-major order using the local facet
/// numbering from get_entity_vertices. Each row has the form [v0, ..., v_{n-1},
/// -1, -1], where v_i are the sorted vertex global indices of the facets and -1
/// is a padding value for the mixed topology case where facets can have
/// differing number of vertices.
/// 3. The number of columns in the unmatched facets array, i.e. the maximum
/// number of vertices per facet.
/// 4. The attached cell (local index) for each unmatched facet in the unmatched
/// facets array.
/// 5. Edge weights aligned with the local dual graph's array().
/// 6. Original cell-side weights aligned with the returned unmatched facets.
/// Both weight vectors are empty in the unweighted case. Unmatched weights
/// remain unaveraged so later distributed matching can include remote cells.
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>, int,
           std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>>
build_local_dual_graph(
    std::span<const CellType> celltypes,
    const std::vector<std::span<const std::int64_t>>& cells,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
    std::span<const std::span<const std::int32_t>> facet_weights);

/// @brief Build distributed mesh dual graph (cell-cell connections via
/// facets) from minimal mesh data.
///
/// The computed dual graph is typically passed to a graph partitioner.
///
/// @note Collective function.
///
/// @param[in] comm The MPI communicator
/// @param[in] celltypes List of cell types
/// @param[in] cells Collections of cells, defined by the cell vertices
/// from which to build the dual graph, as flattened arrays for each
/// cell type in `celltypes`.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet needs to be connected to be considered *matched*, i.e. a
/// matched facet is not connected any cells on other processes. All
/// facets connected to less than `max_facet_to_cell_links` cells are
/// considered *unmatched* and parallel communication will check for
/// further connections. Defaults to `2`, which covers non-branching
/// manifold meshes. Passing std::nullopt (no upper bound) corresponds
/// to `max_facet_to_cell_links`=∞, i.e. every facet is considered
/// unmatched.
/// @param[in] num_threads Number of threads to use. Must be greater
/// than 0.
///
/// @return The dual graph.
///
/// @note `cells` and `celltypes` must have the same size.
///
/// @note The assumption in `build_local_dual_graph` on how unmatched
/// facets are identified will not allow for T-joints (or any other
/// higher branching) across process boundaries to be picked up by the
/// dual graph. If the joints do not live on the process boundary this
/// is not a problem.
graph::AdjacencyList<std::int64_t>
build_dual_graph(MPI_Comm comm, std::span<const CellType> celltypes,
                 const std::vector<std::span<const std::int64_t>>& cells,
                 std::optional<std::int32_t> max_facet_to_cell_links,
                 int num_threads = 1);

} // namespace dolfinx::mesh
