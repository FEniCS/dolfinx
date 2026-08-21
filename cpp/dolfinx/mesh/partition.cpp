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
    mesh::GhostMode ghost_mode, graph::partition_fn partfn,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads)
{
  if (num_threads < 1)
    throw std::runtime_error("num_threads must be >= 1.");

  return [partfn = std::move(partfn), ghost_mode, max_facet_to_cell_links,
          num_threads](MPI_Comm comm, int nparts,
                       const std::vector<CellType>& cell_types,
                       const std::vector<std::span<const std::int64_t>>& cells)
             -> graph::AdjacencyList<std::int32_t>
  {
    spdlog::info("Compute partition of cells across ranks");

    // Compute distributed dual graph (for the cells on this process)
    graph::AdjacencyList dual_graph = build_dual_graph(
        comm, cell_types, cells, max_facet_to_cell_links, num_threads);

    // Just flag any kind of ghosting for now
    bool ghosting = (ghost_mode != GhostMode::none);

    // Compute partition
    return partfn(comm, nparts, dual_graph, ghosting);
  };
}
//-----------------------------------------------------------------------------
mesh::CellPartitionFunction mesh::create_cell_partitioner(
    mesh::GhostMode ghost_mode,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads)
{
  return create_cell_partitioner(ghost_mode, graph::partition_graph,
                                 max_facet_to_cell_links, num_threads);
}
//-----------------------------------------------------------------------------
mesh::GeometricCellPartitionFunction mesh::create_geometric_cell_partitioner(
    mesh::GhostMode ghost_mode,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
    graph::geom_partition_fn partfn)
{
  return [ghost_mode, max_facet_to_cell_links, num_threads,
          partfn = std::move(partfn)](
             MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
             const std::vector<std::span<const std::int64_t>>& cells,
             MPI_Comm /*commg*/, std::span<const double> x,
             std::array<std::size_t, 2> xshape)
             -> graph::AdjacencyList<std::int32_t>
  {
    spdlog::info("Compute geometric partition of cells across ranks");
    common::Timer timer("Compute geometric partition of cells");

    // The mesh dual graph is required to determine ghost cells, and a
    // partitioner may also use it (graph::sfc::partitioner does not)
    if (ghost_mode != GhostMode::none)
    {
      graph::AdjacencyList g = build_dual_graph(
          comm, cell_types, cells, max_facet_to_cell_links, num_threads);
      return partfn(comm, nparts, g, x, xshape[1], true);
    }
    else
    {
      return partfn(comm, nparts, std::nullopt, x, xshape[1], false);
    }
  };
}
//-----------------------------------------------------------------------------
mesh::GeometricCellPartitionFunction mesh::create_hybrid_cell_partitioner(
    mesh::GhostMode ghost_mode,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
    graph::hybrid_partition_fn partfn)
{
  return [ghost_mode, max_facet_to_cell_links, num_threads,
          partfn = std::move(partfn)](
             MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
             const std::vector<std::span<const std::int64_t>>& cells,
             MPI_Comm /*commg*/, std::span<const double> x,
             std::array<std::size_t, 2> /*xshape*/)
             -> graph::AdjacencyList<std::int32_t>
  {
    spdlog::info("Compute hybrid partition of cells across ranks");
    common::Timer timer("Compute hybrid partition of cells");

    // A hybrid partitioner needs the graph edges as part of the
    // partitioning decision itself, not only for ghosting, so the dual
    // graph is always built.
    graph::AdjacencyList g = build_dual_graph(
        comm, cell_types, cells, max_facet_to_cell_links, num_threads);
    return partfn(comm, nparts, g, x, ghost_mode != GhostMode::none);
  };
}
//-----------------------------------------------------------------------------
