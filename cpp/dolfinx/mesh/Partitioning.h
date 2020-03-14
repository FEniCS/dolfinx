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

namespace dolfinx
{

namespace mesh
{

enum class CellType;
class Topology;

/// Tools for partitioning meshes

class Partitioning
{
public:
  /// @todo Move elsewhere
  ///
  /// Compute markers for interior/boundary vertices
  /// @param[in] topology_local Local topology
  /// @return Array where the ith entry is true if the ith vertex is on
  ///   the boundary
  static std::vector<bool>
  compute_vertex_exterior_markers(const mesh::Topology& topology_local);

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
  /// @return Destination processes for each cell on this process
  static graph::AdjacencyList<std::int32_t>
  partition_cells(MPI_Comm comm, int n, const mesh::CellType& cell_type,
                  const graph::AdjacencyList<std::int64_t>& cells);
};
} // namespace mesh
} // namespace dolfinx
