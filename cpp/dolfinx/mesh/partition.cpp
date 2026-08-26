// Copyright (C) 2019-2026 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "partition.h"
#include "graphbuild.h"
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
mesh::CellPartitionFunction mesh::create_cell_partitioner(
    graph::partition_fn partfn,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads)
{
  if (num_threads < 1)
    throw std::runtime_error("num_threads must be >= 1.");

  return [partfn = std::move(partfn), max_facet_to_cell_links, num_threads](
             MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
             const std::vector<std::span<const std::int64_t>>& cells,
             std::span<const std::int32_t> cell_weights,
             std::span<const std::int32_t> edge_weights,
             bool ghosting) -> graph::AdjacencyList<std::int32_t>
  {
    spdlog::info("Compute partition of cells across ranks");

    // Compute distributed dual graph (for the cells on this process)
    graph::AdjacencyList dual_graph = build_dual_graph(
        comm, cell_types, cells, max_facet_to_cell_links, num_threads);

    // Compute partition
    return partfn(comm, nparts, dual_graph, cell_weights, edge_weights,
                  ghosting);
  };
}
//-----------------------------------------------------------------------------
mesh::CellPartitionFunction mesh::create_cell_partitioner(
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads)
{
  return create_cell_partitioner(graph::partition_graph,
                                 max_facet_to_cell_links, num_threads);
}
//-----------------------------------------------------------------------------
mesh::GeometricPartitionFunction
mesh::create_geometric_cell_partitioner(graph::geom_partition_fn partfn)
{
  return [partfn = std::move(partfn)](
             MPI_Comm comm, int nparts, MPI_Comm /*commg*/,
             std::span<const double> x,
             std::array<std::size_t, 2> xshape) -> std::vector<int>
  {
    spdlog::info("Compute geometric partition of cells across ranks");
    common::Timer timer("Compute geometric partition of cells");

    // No cell topology is available to this function, so no dual graph
    // can be built: never ghost.
    return partfn(comm, nparts, x, xshape[1]);
  };
}
//-----------------------------------------------------------------------------
mesh::HybridCellPartitionFunction mesh::create_hybrid_cell_partitioner(
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
    graph::hybrid_partition_fn partfn)
{
  return [max_facet_to_cell_links, num_threads, partfn = std::move(partfn)](
             MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
             const std::vector<std::span<const std::int64_t>>& cells,
             MPI_Comm /*commg*/, std::span<const double> x,
             std::array<std::size_t, 2> /*xshape*/,
             bool ghosting) -> graph::AdjacencyList<std::int32_t>
  {
    spdlog::info("Compute hybrid partition of cells across ranks");
    common::Timer timer("Compute hybrid partition of cells");

    // A hybrid partitioner needs the graph edges as part of the
    // partitioning decision itself, not only for ghosting, so the dual
    // graph is always built.
    graph::AdjacencyList g = build_dual_graph(
        comm, cell_types, cells, max_facet_to_cell_links, num_threads);
    return partfn(comm, nparts, g, x, ghosting);
  };
}
//-----------------------------------------------------------------------------
