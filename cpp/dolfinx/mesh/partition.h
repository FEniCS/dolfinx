// Copyright (C) 2019-2026 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include "graphbuild.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/sfc.h>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <variant>
#include <vector>

/// @file partition.h
/// @brief Cell partitioning function types, and functions that create
/// them.

namespace dolfinx::mesh
{
/// Enum for different partitioning ghost modes
enum class GhostMode : std::uint8_t
{
  none,
  shared_facet
};

/// @brief Signature for the cell partitioning function. Functions that
/// implement this interface compute the destination rank for cells
/// currently on this rank.
///
/// @param[in] comm MPI Communicator.
/// @param[in] nparts Number of partitions.
/// @param[in] cell_types Cell types in the mesh.
/// @param[in] cells Lists of cells of each cell type. `cells[i]` is a
/// flattened row major 2D array of shape (num_cells, num_cell_vertices)
/// for `cell_types[i]` on this process, containing the global indices
/// for the cell vertices. Each cell can appear only once across all
/// processes. The cell vertex indices are not necessarily contiguous
/// globally, i.e. the maximum index across all processes can be greater
/// than the number of vertices. High-order 'nodes', e.g. mid-side
/// points, should not be included.
/// @return Destination rank(s) for each cell on this process, the
/// owning rank first. A cell has more than one destination rank only
/// when it is ghosted.
using CellPartitionFunction = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
    const std::vector<std::span<const std::int64_t>>& cells)>;

/// @brief Signature for a cell partitioning function that also has
/// access to the cell centroids, e.g. to partition using them rather
/// than (or in addition to) the mesh dual graph.
///
/// As ::CellPartitionFunction, with the addition of the cell
/// centroids, `x`. `x` holds one row per cell in `cells` (in the same
/// order), not the mesh 'node' coordinates -- see ::AnyCellPartitionFunction
/// and ::create_geometric_cell_partitioner for how the centroids are
/// obtained. Since the centroids are already local to the cells on
/// this rank, `commg` is typically unused by an implementation.
///
/// @note The coordinates are always `double`, whatever the scalar type
/// of the mesh being created, for the same reason as
/// graph::partition_fn: partitioning is not sensitive to the precision
/// of the positions.
///
/// @param[in] comm MPI Communicator that `cell_types`/`cells` are
/// distributed across.
/// @param[in] nparts Number of partitions.
/// @param[in] cell_types Cell types in the mesh.
/// @param[in] cells Lists of cells of each cell type, as
/// ::CellPartitionFunction.
/// @param[in] commg MPI Communicator that `x` is distributed across.
/// @param[in] x Cell centroids, row-major with `xshape[1]` columns,
/// one row per cell across `cells` (in the same order).
/// @param[in] xshape Shape of `x`.
/// @return Destination rank(s) for each cell on this process, as
/// ::CellPartitionFunction.
using GeometricCellPartitionFunction
    = std::function<graph::AdjacencyList<std::int32_t>(
        MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
        const std::vector<std::span<const std::int64_t>>& cells, MPI_Comm commg,
        std::span<const double> x, std::array<std::size_t, 2> xshape)>;

/// @brief Either kind of cell partitioning function that ::create_mesh
/// accepts.
///
/// ::create_mesh always has the cell topology available, so a
/// ::CellPartitionFunction alternative is simply called directly. If
/// the alternative held is a ::GeometricCellPartitionFunction,
/// ::create_mesh first computes the centroid of each cell from the
/// vertex coordinates -- using the same `(commg, x, xshape)` geometry
/// data it uses to build the mesh -- and supplies them, since a
/// GeometricCellPartitionFunction has no other way to obtain them.
using AnyCellPartitionFunction
    = std::variant<CellPartitionFunction, GeometricCellPartitionFunction>;

/// @brief Create a function that computes destination rank for mesh
/// cells on this rank by applying `partfn` to the dual graph of the
/// mesh.
///
/// @param[in] ghost_mode Ghost mode of the created mesh.
/// @param[in] partfn Partitioning function for distributing cells
/// across MPI ranks.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet must be connected to for it to be considered *matched* (not
/// on boundary for non-branching meshes).
/// @param[in] num_threads Number of threads to use when building the
/// dual graph. Must be >= 1.
/// @return Function that computes the destination ranks for each cell.
CellPartitionFunction
create_cell_partitioner(mesh::GhostMode ghost_mode, graph::partition_fn partfn,
                        std::optional<std::int32_t> max_facet_to_cell_links,
                        int num_threads = 1);

/// @brief Create a function that computes destination rank for mesh
/// cells on this rank by applying the default graph partitioner to the
/// dual graph of the mesh.
///
/// @param[in] ghost_mode Ghost mode of the created mesh.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet must be connected to for it to be considered *matched* (not
/// on boundary for non-branching meshes).
/// @param[in] num_threads Number of threads to use when building the
/// dual graph. Must be >= 1.
/// @return Function that computes the destination ranks for each cell.
CellPartitionFunction
create_cell_partitioner(mesh::GhostMode ghost_mode,
                        std::optional<std::int32_t> max_facet_to_cell_links,
                        int num_threads = 1);

namespace impl
{
/// @brief Compute the centroid of each cell from its vertex positions.
///
/// @tparam T Scalar type of `x`.
/// @param[in] comm Communicator that `cells` is distributed across.
/// @param[in] num_vertices_per_cell Number of vertices per cell, one
/// entry per cell type of `cells`.
/// @param[in] cells Cells of each cell type, using global vertex
/// indices (no higher-order 'nodes').
/// @param[in] commg Communicator that `x` is distributed across.
/// @param[in] x Geometry ('node') coordinates, row-major with `gdim`
/// columns, distributed over `commg`.
/// @param[in] gdim Number of coordinate components per node.
/// @return Cell centroids, row-major with `gdim` columns, one row per
/// cell, with the cells of each cell type concatenated in the order
/// they appear in `cells`.
template <std::floating_point T>
std::vector<double>
compute_cell_centroids(MPI_Comm comm,
                       std::span<const int> num_vertices_per_cell,
                       const std::vector<std::span<const std::int64_t>>& cells,
                       MPI_Comm commg, std::span<const T> x, int gdim)
{
  // Vertices of the cells on this rank, sorted and with duplicates
  // removed, and the coordinates for them
  std::vector<std::int64_t> nodes;
  {
    std::size_t size = 0;
    for (std::span<const std::int64_t> c : cells)
      size += c.size();
    nodes.reserve(size);
    for (std::span<const std::int64_t> c : cells)
      nodes.insert(nodes.end(), c.begin(), c.end());
    dolfinx::radix_sort(nodes);
    auto [unique_end, range_end] = std::ranges::unique(nodes);
    nodes.erase(unique_end, range_end);
  }
  const std::vector<T> coords
      = dolfinx::MPI::distribute_data(comm, nodes, commg, x, gdim);

  // Cell 'centroids', i.e. the mean of the cell vertex positions
  std::size_t num_cells = 0;
  for (std::size_t i = 0; i < cells.size(); ++i)
    num_cells += cells[i].size() / num_vertices_per_cell[i];
  std::vector<double> centroid(gdim * num_cells, 0);

  std::size_t c0 = 0;
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    const int nv = num_vertices_per_cell[i];
    const double w = 1.0 / nv;
    for (std::size_t c = 0; c < cells[i].size() / nv; ++c)
    {
      for (int v = 0; v < nv; ++v)
      {
        // Position of the vertex in `nodes` is its row in `coords`
        auto it = std::ranges::lower_bound(nodes, cells[i][nv * c + v]);
        assert(it != nodes.end() and *it == cells[i][nv * c + v]);
        std::size_t pos = std::ranges::distance(nodes.begin(), it);
        for (int d = 0; d < gdim; ++d)
          centroid[gdim * (c0 + c) + d] += w * coords[gdim * pos + d];
      }
    }

    c0 += cells[i].size() / nv;
  }

  return centroid;
}
} // namespace impl

/// @brief Create a function that computes the destination rank for mesh
/// cells on this rank from the position of the cell 'centroids' on a
/// space-filling curve (see graph::sfc::partitioner).
///
/// Unlike ::create_cell_partitioner, no graph partitioner is used by
/// default (though `partfn` may use one). This is markedly cheaper --
/// the cost is nearly independent of the number of MPI ranks, whereas
/// graph partitioning cost grows with the rank count -- and the load
/// balance is near-perfect, but more cells lie on the partition
/// boundary, i.e. there are more ghost cells and more communication in
/// the created mesh. It is intended for cases where mesh partitioning
/// cost dominates, e.g. very large rank counts.
///
/// @note The returned ::GeometricCellPartitionFunction is called with
/// the cell *centroids* as its `x` argument, not the mesh node
/// coordinates -- e.g. as ::create_mesh supplies automatically when
/// this is used as (part of) an ::AnyCellPartitionFunction. The
/// `commg`/`xshape` it is called with describe that centroid array, and
/// are unused here since the centroids are already local to the
/// calling rank.
///
/// @param[in] ghost_mode Ghost mode of the created mesh. Cell ghosting
/// requires the mesh dual graph, which is built only if `ghost_mode` is
/// not GhostMode::none.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet needs to be connected to to be considered *matched* (not on
/// boundary for non-branching meshes).
/// @param[in] num_threads Number of threads to use when building the
/// dual graph (cell ghosting only). Must be >= 1.
/// @param[in] partfn Geometric graph partitioner to apply to the cell
/// centroids. Defaults to the space-filling curve partitioner,
/// graph::sfc::partitioner. The centroids are always supplied; the dual
/// graph is supplied only when `ghost_mode` is not GhostMode::none, and
/// is otherwise absent (see ::geom_partition_fn).
/// @return A geometric cell partitioning function.
GeometricCellPartitionFunction create_geometric_cell_partitioner(
    mesh::GhostMode ghost_mode,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads = 1,
    graph::geom_partition_fn partfn = graph::sfc::partitioner());

/// @brief Create a function that computes the destination rank for mesh
/// cells on this rank using a hybrid partitioner that needs both the
/// mesh dual graph and the cell 'centroids' (e.g. ParMETIS `GeomKway`,
/// see graph::parmetis::geom_partitioner_kway).
///
/// Unlike ::create_geometric_cell_partitioner, the dual graph is always
/// built and supplied to `partfn`, regardless of `ghost_mode`, since a
/// hybrid partitioner uses the graph edges as part of the partitioning
/// decision itself, not only for ghosting.
///
/// @param[in] ghost_mode Ghost mode of the created mesh.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet needs to be connected to to be considered *matched* (not on
/// boundary for non-branching meshes).
/// @param[in] num_threads Number of threads to use when building the
/// dual graph. Must be >= 1.
/// @param[in] partfn Hybrid graph partitioner to apply to the dual graph
/// and the cell centroids.
/// @return A geometric cell partitioning function.
GeometricCellPartitionFunction create_hybrid_cell_partitioner(
    mesh::GhostMode ghost_mode,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
    graph::hybrid_partition_fn partfn);

} // namespace dolfinx::mesh
