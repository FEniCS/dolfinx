// Copyright (C) 2010-2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Graph.h"
#include <boost/multi_array.hpp>
#include <boost/unordered_map.hpp>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <set>
#include <utility>
#include <vector>

namespace dolfin
{
namespace fem
{
class GenericDofMap;
}

namespace mesh
{
class CellType;
class Mesh;
}

namespace graph
{

/// This class builds a Graph corresponding to various objects

class GraphBuilder
{

public:
  /// Build local graph from dofmap
  static Graph local_graph(const mesh::Mesh& mesh,
                           const fem::GenericDofMap& dofmap0,
                           const fem::GenericDofMap& dofmap1);

  /// Build local graph from mesh (general version)
  static Graph local_graph(const mesh::Mesh& mesh,
                           const std::vector<std::size_t>& coloring_type);

  /// Build local graph (specialized version)
  static Graph local_graph(const mesh::Mesh& mesh, std::size_t dim0,
                           std::size_t dim1);

  /// Build distributed dual graph (cell-cell connections) from
  /// minimal mesh data, and return (num local edges, num
  /// non-local edges)
  static std::pair<std::int32_t, std::int32_t>
  compute_dual_graph(const MPI_Comm mpi_comm,
                     const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
                     const mesh::CellType& cell_type,
                     std::vector<std::vector<std::size_t>>& local_graph,
                     std::set<std::int64_t>& ghost_vertices);

private:
  friend class mesh::MeshPartitioning;

  typedef std::vector<std::pair<std::vector<std::size_t>, std::int32_t>>
      FacetCellMap;

  // Compute local part of the dual graph, and return number of
  // local edges in the graph (undirected)
  static std::int32_t compute_local_dual_graph(
      const MPI_Comm mpi_comm,
      const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
      const mesh::CellType& cell_type,
      std::vector<std::vector<std::size_t>>& local_graph,
      FacetCellMap& facet_cell_map);

  // Compute local part of the dual graph, and return number of
  // local edges in the graph (undirected)
  template <int N>
  static std::int32_t compute_local_dual_graph_keyed(
      const MPI_Comm mpi_comm,
      const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
      const mesh::CellType& cell_type,
      std::vector<std::vector<std::size_t>>& local_graph,
      FacetCellMap& facet_cell_map);

  // Build nonlocal part of dual graph for mesh and return number of
  // non-local edges. Note: GraphBuilder::compute_local_dual_graph
  // should be called before this function is called.
  static std::int32_t compute_nonlocal_dual_graph(
      const MPI_Comm mpi_comm,
      const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
      const mesh::CellType& cell_type,
      std::vector<std::vector<std::size_t>>& local_graph,
      FacetCellMap& facet_cell_map, std::set<std::int64_t>& ghost_vertices);
};
}
}
