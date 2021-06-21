// Copyright (C) 2010-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <utility>
#include <xtensor/xarray.hpp>

namespace dolfinx::mesh
{

enum class CellType;

/// Build distributed dual graph (cell-cell connections) from minimal
/// mesh data, and return (graph, ghost_vertices, [num local edges,
/// num non-local edges])
std::pair<graph::AdjacencyList<std::int64_t>, std::array<std::int32_t, 2>>
build_dual_graph(const MPI_Comm comm,
                 const graph::AdjacencyList<std::int64_t>& cell_vertices,
                 int tdim);

/// Compute local part of the dual graph (cell-cell connections via
/// facets)
///
/// @todo Fix return description - it's unclear
///
/// @param[in] cell_vertices Array for cell vertices
/// @param[in] offsets Index of the first entry of cell `i` in
/// `cell_vertices` `is offsets[i]`
/// @param[in] tdim The topological dimension if the cells
/// @return (local_graph, facet_cell_map, number of local edges in the
/// graph (undirected)
std::pair<graph::AdjacencyList<std::int32_t>, xt::xtensor<std::int64_t, 2>>
build_local_dual_graph(const xtl::span<const std::int64_t>& cell_vertices,
                       const xtl::span<const std::int32_t>& offsets, int tdim);

} // namespace dolfinx::mesh
