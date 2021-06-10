// Copyright (C) 2007 Kristian B. Oelgaard
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>

namespace dolfinx
{
namespace fem
{
class CoordinateElement;
}

/// Interval mesh creation
namespace generation::IntervalMesh
{

/// Interval mesh of the 1D line `[a, b]`.  Given @p n cells in the
/// axial direction, the total number of intervals will be `n` and the
/// total number of vertices will be `n + 1`.
///
/// @param[in] comm MPI communicator to build the mesh on
/// @param[in] n The number of cells
/// @param[in] x The end points of the interval
/// @param[in] ghost_mode Ghosting mode
/// @param[in] partitioner Partitioning function to use for determining
/// the parallel distribution of cells across MPI ranks
/// @return A mesh
mesh::Mesh
create(MPI_Comm comm, std::size_t n, std::array<double, 2> x,
       const mesh::GhostMode ghost_mode,
       const mesh::CellPartitionFunction& partitioner
       = static_cast<graph::AdjacencyList<std::int32_t> (*)(
           MPI_Comm, int, int, const graph::AdjacencyList<std::int64_t>&,
           mesh::GhostMode)>(&mesh::partition_cells_graph));
} // namespace generation::IntervalMesh
} // namespace dolfinx
