// Copyright (C) 2010-2013 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <tuple>
#include <utility>
#include <vector>

namespace dolfinx::mesh
{

enum class CellType;

/// Tools to build a Graph corresponding to various objects
namespace GraphBuilder
{

/// Build distributed dual graph (cell-cell connections) from minimal
/// mesh data, and return (graph, ghost_vertices, [num local edges,
/// num non-local edges])
std::pair<std::vector<std::vector<std::int64_t>>, std::array<std::int32_t, 3>>
compute_dual_graph(const MPI_Comm mpi_comm,
                   const graph::AdjacencyList<std::int64_t>& cell_vertices,
                   const mesh::CellType& cell_type);

/// Compute local part of the dual graph, and return (local_graph,
/// facet_cell_map, number of local edges in the graph (undirected)
std::tuple<std::vector<std::vector<std::int32_t>>,
           std::vector<std::pair<std::vector<std::int32_t>, std::int32_t>>,
           std::int32_t>
compute_local_dual_graph(
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type);

} // namespace GraphBuilder
} // namespace dolfinx::mesh
