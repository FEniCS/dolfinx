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

namespace dolfinx
{

namespace mesh
{

enum class CellType;

/// This class builds a Graph corresponding to various objects

class GraphBuilder
{

public:
  /// Build distributed dual graph (cell-cell connections) from minimal
  /// mesh data, and return (graph, ghost_vertices, [num local edges,
  /// num non-local edges])
  static std::pair<std::vector<std::vector<std::int64_t>>,
                   std::array<std::int32_t, 3>>
  compute_dual_graph(
      const MPI_Comm mpi_comm,
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          cell_vertices,
      const mesh::CellType& cell_type);

  /// Compute local part of the dual graph, and return (local_graph,
  /// facet_cell_map, number of local edges in the graph (undirected)
  static std::tuple<
      std::vector<std::vector<std::int32_t>>,
      std::vector<std::pair<std::vector<std::int32_t>, std::int32_t>>,
      std::int32_t>
  compute_local_dual_graph(
      const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          cell_vertices,
      const mesh::CellType& cell_type);
};
} // namespace mesh
} // namespace dolfinx
