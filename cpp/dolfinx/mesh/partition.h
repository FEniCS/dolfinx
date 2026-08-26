// Copyright (C) 2019-2026 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/partition.h>
#include <iterator>
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

/// @brief Any kind of cell partitioning function that ::create_mesh
/// accepts: a graph::partition_fn, a graph::geom_partition_fn, or a
/// graph::hybrid_partition_fn.
///
/// ::create_mesh always has the cell topology available, so it can
/// build the mesh dual graph itself and pass it to a
/// graph::partition_fn or a graph::hybrid_partition_fn alternative,
/// neither of which has any other way to obtain it. If the alternative
/// held is a graph::geom_partition_fn or a graph::hybrid_partition_fn,
/// ::create_mesh additionally computes the centroid of each cell from
/// the vertex coordinates -- using the same `(commg, x, xshape)`
/// geometry data it uses to build the mesh -- and supplies them, since
/// neither has any other way to obtain them.
using AnyCellPartitionFunction
    = std::variant<graph::partition_fn, graph::geom_partition_fn,
                   graph::hybrid_partition_fn>;

namespace impl
{
/// @brief Compute the centroid of each cell from its vertex positions.
///
/// @tparam T Scalar type of `x`.
/// @param[in] comm Communicator that `cells` is distributed across.
/// @param[in] num_vertices_per_cell Number of vertices per cell, one
/// entry per cell type of `cells`.
/// @param[in] cells Cells of each cell type, using global vertex
/// indices (no higher-order 'nodes'). `cells[i]` is a flattened
/// row-major array of shape `(num_cells_i, num_vertices_per_cell[i])`
/// for cell type `i`, where `num_cells_i` is however many cells of
/// that type are on this rank.
/// @param[in] commg Communicator that `x` is distributed across.
/// @param[in] x Geometry ('node') coordinates, row-major with `gdim`
/// columns, distributed over `commg`. Rows are addressed by the
/// global vertex indices used in `cells`; only the rows for vertices
/// referenced by `cells` on this rank are gathered from `commg`.
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

} // namespace dolfinx::mesh
