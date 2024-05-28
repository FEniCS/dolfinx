// Copyright (C) 2010-2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <mpi.h>
#include <span>
#include <tuple>
#include <vector>

namespace dolfinx::graph
{
template <typename T>
class AdjacencyList;
}

namespace dolfinx::mesh
{
enum class CellType;

/// @brief Compute the local part of the dual graph (cell-cell
/// connections via facets) and facets with only one attached cell.
///
/// @param[in] celltypes List of cell types.
/// @param[in] cells Lists of cell vertices (stored as flattened lists for each
/// cell type).
/// @return
/// 1. Local dual graph
/// 2. Facets, defined by their vertices, that are shared by only one
/// cell on this rank. The logically 2D array is flattened (row-major).
/// 3. The number of columns for the facet data array (2).
/// 4. The attached cell (local index) to each returned facet in (2).
///
/// Each row of the returned data (2) contains `[v0, ... v_(n-1), x, ..,
/// x]`, where `v_i` is a vertex global index, `x` is a padding value
/// (all padding values will be equal).
///
/// @note The cell indexing stacks the cells from each cell type consecutively,
/// i.e. if there are n cells of type 0 and m cells of type 1, then cells are
/// numbered 0..(n-1) and n..(n+m-1) respectively in the dual graph.
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>,
           std::size_t, std::vector<std::int32_t>>
build_local_dual_graph(std::vector<CellType> celltypes,
                       std::vector<std::span<const std::int64_t>> cells);

/// @brief Build distributed mesh dual graph (cell-cell connections via
/// facets) from minimal mesh data.
///
/// The computed dual graph is typically passed to a graph partitioner.
///
/// @note Collective function
///
/// @param[in] comm The MPI communicator
/// @param[in] celltypes List of cell types
/// @param[in] cells Collections of cells, defined by the cell vertices
/// from which to build the dual graph, as flattened arrays for each cell type.
/// @return The dual graph
graph::AdjacencyList<std::int64_t>
build_dual_graph(MPI_Comm comm, const std::vector<CellType>& celltypes,
                 const std::vector<std::span<const std::int64_t>>& cells);

} // namespace dolfinx::mesh
