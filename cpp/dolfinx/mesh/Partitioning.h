// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <vector>

namespace dolfinx::mesh
{

enum class CellType;
class Topology;
enum class GhostMode : int;

/// Tools for partitioning meshes
namespace Partitioning
{
/// Compute destination rank for mesh cells in this rank using a graph
/// partitioner
///
/// @param[in] comm MPI Communicator
/// @param[in] n Number of partitions
/// @param[in] cell_type Cell type
/// @param[in] cells Cells on this process. The ith entry in list
///   contains the global indices for the cell vertices. Each cell can
///   appear only once across all processes. The cell vertex indices
///   are not necessarily contiguous globally, i.e. the maximum index
///   across all processes can be greater than the number of vertices.
///   High-order 'nodes', e.g. mid-side points, should not be
///   included.
/// @param[in] ghost_mode How to overlap the cell partitioning: none,
///   shared_facet or shared_vertex
/// @return Destination processes for each cell on this process
graph::AdjacencyList<std::int32_t>
partition_cells(MPI_Comm comm, int n, const mesh::CellType cell_type,
                const graph::AdjacencyList<std::int64_t>& cells,
                mesh::GhostMode ghost_mode);

} // namespace Partitioning
} // namespace dolfinx::mesh
