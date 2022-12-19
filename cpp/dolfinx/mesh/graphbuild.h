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
/// @brief Compute the local part of the dual graph (cell-cell
/// connections via facets) and facet with only one attached cell.
///
/// @param[in] cells Array for cell vertices adjacency list
/// @param[in] offsets Adjacency list offsets, i.e. index of the first
/// entry of cell `i` in `cell_vertices` `is offsets[i]`
/// @param[in] tdim The topological dimension of the cells
/// @return
/// 1. Local dual graph
/// 2. Facets, defined by their vertices, that are shared by only one
/// cell on this rank. The logically 2D is array flattened (row-major).
/// 3. The number of columns for the facet data array (2).
/// 4. The attached cell (local index) to each returned facet in (2).
///
/// Each row of the returned data (2) contains `[v0, ... v_(n-1), x, ..,
/// x]`, where `v_i` is a vertex global index, `x` is a padding value
/// (all padding values will be equal).
///
/// @note The return data will likely change once we support mixed
/// topology meshes.
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>,
           std::size_t, std::vector<std::int32_t>>
build_local_dual_graph(std::span<const std::int64_t> cells,
                       std::span<const std::int32_t> offsets, int tdim);

/// @brief Build distributed mesh dual graph (cell-cell connections via
/// facets) from minimal mesh data.
///
/// The computed dual graph is typically passed to a graph partitioner.
///
/// @note Collective function
///
/// @param[in] comm The MPI communicator
/// @param[in] cells Collection of cells, defined by the cell vertices
/// from which to build the dual graph
/// @param[in] tdim The topological dimension of the cells
/// @return The dual graph
graph::AdjacencyList<std::int64_t>
build_dual_graph(const MPI_Comm comm,
                 const graph::AdjacencyList<std::int64_t>& cells, int tdim);

} // namespace dolfinx::mesh
