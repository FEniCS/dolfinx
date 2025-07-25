// Copyright (C) 2010-2025 Garth N. Wells and Paul T. Kühner
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

/// @brief Compute the local part of the dual graph (cell-cell
/// connections via facets) and facets with only one attached cell.
///
/// @param[in] celltypes List of cell types.
/// @param[in] cells Lists of cell vertices (stored as flattened lists,
/// one for each cell type).
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet needs to be connected to to be considered *matched*, i.e. a
/// matched facet is not connected any cells on other processes. All
/// facets connected to less than `max_facet_to_cell_links` cells are
/// considered *unmatched* and parallel communication will check for
/// further connections. Defaults to `2`, which covers non-branching
/// manifold meshes. Passing std::nullopt (no upper bound) corresponds
/// to `max_facet_to_cell_links`=∞, i.e. every facet is considered
/// unmatched.
///
/// @return
/// 1. Local dual graph
/// 2. Facets, defined by their sorted vertices, that are shared by only
///   `max_facet_to_cell_links` or less cells on this rank. The logically
///   2D array is flattened (row-major).
/// 3. Facet data array (2) number of columns
/// 4. Attached cell (local index) to each returned facet in (2).
///
/// Each row of the returned data (2) contains `[v0, ... v_(n-1), x, ..,
/// x]`, where `v_i` is a vertex global index, `x` is a negative value
/// (all padding values will be equal). The vertex global indices are
/// sorted for each facet.
///
/// @note The cells of each cell type are numbered locally
/// consecutively, i.e. if there are `n` cells of type `0` and `m` cells
/// of type `1`, then cells of type `0` are numbered `0..(n-1)` and
/// cells of type `1` are numbered `n..(n+m-1)` respectively, in the
/// returned dual graph.
///
/// @note Facet (2) and cell (4) data will contain multiple entries for
/// the same facet for branching meshes with `max_facet_to_cell_links>2`
/// to account for all facet cell connectivies.
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>,
           std::size_t, std::vector<std::int32_t>>
build_local_dual_graph(std::span<const CellType> celltypes,
                       const std::vector<std::span<const std::int64_t>>& cells,
                       std::optional<std::int32_t> max_facet_to_cell_links = 2);

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
/// facet needs to be connected to to be considered *matched*, i.e. a
/// matched facet is not connected any cells on other processes. All
/// facets connected to less than `max_facet_to_cell_links` cells are
/// considered *unmatched* and parallel communication will check for
/// further connections. Defaults to `2`, which covers non-branching
/// manifold meshes. Passing std::nullopt (no upper bound) corresponds
/// to `max_facet_to_cell_links`=∞, i.e. every facet is considered
/// unmatched.
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
                 std::optional<std::int32_t> max_facet_to_cell_links = 2);

} // namespace dolfinx::mesh
