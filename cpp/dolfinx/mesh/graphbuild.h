// Copyright (C) 2010-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <utility>
#include <xtensor/xarray.hpp>

namespace dolfinx::mesh
{

/// Compute the local part of the dual graph (cell-cell connections via
/// facets)
///
/// @param[in] cells Array for cell vertices adjacency list
/// @param[in] offsets Adjacency list offsets, i.e. index of the first
/// entry of cell `i` in `cell_vertices` `is offsets[i]`
/// @param[in] tdim The topological dimension of the cells
/// @return (0) Local dual graph and (1) facet data for facets that are
/// shared by only one cell on this rank. The facet data is `[v0, ...
/// v_(n-1), x, .., x, cell_index]`, where `v_i` is a vertex global
/// index, `x` is a padding value (all padding values will be equal) and
/// `cell_index` is the global index of the attached cell.
std::pair<graph::AdjacencyList<std::int32_t>, xt::xtensor<std::int64_t, 2>>
build_local_dual_graph(const xtl::span<const std::int64_t>& cells,
                       const xtl::span<const std::int32_t>& offsets, int tdim);

/// Build distributed dual graph (cell-cell connections) from minimal
/// mesh data
///
/// @param[in] comm The MPI communicator
/// @param[in] cells Collection of cells, defined by the cell vertices
/// from which to build the dual graph
/// @param[in] tdim The topological dimension of the cells
/// @return The (0) dual graph and (1) number of  ghost edges
/// @note Collective function
std::pair<graph::AdjacencyList<std::int64_t>, std::int32_t>
build_dual_graph(const MPI_Comm comm,
                 const graph::AdjacencyList<std::int64_t>& cells, int tdim);

} // namespace dolfinx::mesh
