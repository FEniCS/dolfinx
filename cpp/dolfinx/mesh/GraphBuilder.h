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
#include <dolfinx/graph/AdjacencyList.h>
#include <tuple>
#include <utility>
#include <vector>

namespace dolfinx::mesh
{

enum class CellType;

/// Build distributed dual graph (cell-cell connections) from minimal
/// mesh data, and return (graph, ghost_vertices, [num local edges,
/// num non-local edges])
std::pair<graph::AdjacencyList<std::int64_t>, std::array<std::int32_t, 3>>
build_dual_graph(const MPI_Comm mpi_comm,
                 const graph::AdjacencyList<std::int64_t>& cell_vertices,
                 const mesh::CellType& cell_type);

/// Compute local part of the dual graph, and return (local_graph,
/// facet_cell_map, number of local edges in the graph (undirected)
std::tuple<graph::AdjacencyList<std::int32_t>,
           std::vector<std::pair<std::vector<std::int64_t>, std::int32_t>>,
           std::int32_t>
build_local_dual_graph(const graph::AdjacencyList<std::int64_t>& cell_vertices,
                       const mesh::CellType& cell_type);

}