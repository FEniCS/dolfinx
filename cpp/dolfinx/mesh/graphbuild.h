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

/// Cell-facet records retained for matching across MPI ranks.
/// A shared facet can have multiple records, one per attached local cell.
struct UnmatchedFacetData
{
  /// Flattened row-major vertex keys, with num_columns entries per row.
  /// Each row contains sorted global vertex indices followed by -1 padding.
  std::vector<std::int64_t> facets;
  /// Number of vertex columns per facet record; zero for empty data.
  int num_columns;
  /// Local cell index for each facet record.
  std::vector<std::int32_t> attached_cells;
  /// Original cell-side weights, one per record when weighted is true.
  /// Not averaged, so remote cell contributions can be included later.
  std::vector<std::int32_t> unmatched_weights;
  /// Whether weights are enabled, independently of the number of records.
  /// Must agree across ranks when used for distributed matching.
  bool weighted;
};

/// @brief Build the local dual graph with optional cell-facet weights.
///
/// @param[in] celltypes List of cell types.
/// @param[in] cells Lists of cell vertices (stored as flattened lists,
/// one for each cell type).
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet needs to be connected to be considered *matched*, i.e. a
/// matched facet is not connected any cells on other processes. All
/// facets connected to less than `max_facet_to_cell_links` cells are
/// considered *unmatched* and parallel communication will check for
/// further connections. Equal to `2` for non-branching manifold meshes.
/// Passing std::nullopt (no upper bound) corresponds
/// to `max_facet_to_cell_links`=∞, i.e. every facet is considered
/// unmatched.
/// @param[in] num_threads Number of threads to use. Must be greater
/// than 0.
/// @param[in] facet_weights Positive integer cell-facet weights, with one
/// array per cell type in `celltypes`. Each array is flattened in cell-major
/// order, with shape `(num_cells, num_facets_per_cell)`. Facet numbering
/// follows `get_entity_vertices(celltypes[j], tdim - 1)`. An empty outer span
/// selects the unweighted path, which allocates no weight buffers.
/// For each shared facet, the mean over all attached cells on this rank is
/// rounded down and assigned to both directions of every cell-pair edge.
///
/// @return
/// 1. Local dual graph.
/// 2. Edge weights aligned with the adjacency entries in the graph's array().
///   Empty when the outer `facet_weights` span is empty.
/// 3. UnmatchedFacetData for facets shared by fewer than
///   `max_facet_to_cell_links` cells on this rank, or all facets if no bound
///   is given. Original cell-side weights are retained for later matching.
///
/// @note The cells of each cell type are numbered locally
/// consecutively, i.e. if there are `n` cells of type `0` and `m` cells
/// of type `1`, then cells of type `0` are numbered `0..(n-1)` and
/// cells of type `1` are numbered `n..(n+m-1)` respectively, in the
/// returned dual graph.
///
/// @note The unmatched data can contain multiple records for the same facet
/// on branching meshes, one for each attached local cell.
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int32_t>,
           UnmatchedFacetData>
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
