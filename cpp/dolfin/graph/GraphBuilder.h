// Copyright (C) 2010-2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Graph.h"
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/cell_types.h>
#include <tuple>
#include <utility>
#include <vector>

namespace dolfin
{
namespace fem
{
class DofMap;
}

namespace mesh
{
class Mesh;
} // namespace mesh

namespace graph
{

/// This class builds a Graph corresponding to various objects

class GraphBuilder
{

public:
  /// Connectivity from facets (defined by their global vertex indices) to cells
  typedef std::vector<std::pair<std::vector<std::size_t>, std::int32_t>>
      FacetCellMap;

  /// Build local graph from dofmap
  static Graph local_graph(const mesh::Mesh& mesh, const fem::DofMap& dofmap0,
                           const fem::DofMap& dofmap1);

  /// Build local graph from mesh (general version)
  static Graph local_graph(const mesh::Mesh& mesh,
                           const std::vector<std::size_t>& coloring_type);

  /// Build local graph (specialized version)
  static Graph local_graph(const mesh::Mesh& mesh, std::size_t dim0,
                           std::size_t dim1);

  /// Build distributed dual graph (cell-cell connections) from
  /// minimal mesh data, and return (graph, ghost_vertices, [num local edges,
  /// num non-local edges])
  static std::pair<std::vector<std::vector<std::size_t>>,
                   std::tuple<std::int32_t, std::int32_t, std::int32_t>>
  compute_dual_graph(const MPI_Comm mpi_comm,
                     const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
                     const mesh::CellType cell_type);

  /// Compute local part of the dual graph, and return (local_graph,
  /// facet_cell_map, number of local edges in the graph (undirected)
  static std::tuple<
      std::vector<std::vector<std::size_t>>,
      std::vector<std::pair<std::vector<std::size_t>, std::int32_t>>,
      std::int32_t>
  compute_local_dual_graph(
      const MPI_Comm mpi_comm,
      const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
      const mesh::CellType cell_type);
};
} // namespace graph
} // namespace dolfin
