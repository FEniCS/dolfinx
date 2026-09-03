// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for the geometric (space-filling curve) cell partitioner

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/graph/sfc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <numeric>
#include <span>
#include <vector>

using namespace dolfinx;

namespace
{
/// @brief Local slice of a structured cube of tetrahedra (six per cube)
/// and of the cube vertex coordinates.
std::pair<std::vector<std::int64_t>, std::vector<double>> cube(MPI_Comm comm,
                                                               std::int64_t n)
{
  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);

  std::vector<std::int64_t> cells;
  std::array<std::int64_t, 2> rc = common::local_range(rank, n * n * n, size);
  for (std::int64_t i = rc[0]; i < rc[1]; ++i)
  {
    const std::int64_t iz = i / (n * n);
    const std::int64_t j = i % (n * n);
    const std::int64_t iy = j / n;
    const std::int64_t ix = j % n;
    const std::int64_t v0 = iz * (n + 1) * (n + 1) + iy * (n + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (n + 1);
    const std::int64_t v3 = v1 + (n + 1);
    const std::int64_t v4 = v0 + (n + 1) * (n + 1);
    const std::int64_t v5 = v1 + (n + 1) * (n + 1);
    const std::int64_t v6 = v2 + (n + 1) * (n + 1);
    const std::int64_t v7 = v3 + (n + 1) * (n + 1);
    cells.insert(cells.end(), {v0, v1, v3, v7, v0, v1, v7, v5, v0, v5, v7, v4,
                               v0, v3, v2, v7, v0, v6, v4, v7, v0, v2, v6, v7});
  }

  std::vector<double> x;
  std::array<std::int64_t, 2> rp
      = common::local_range(rank, (n + 1) * (n + 1) * (n + 1), size);
  const std::int64_t sqxy = (n + 1) * (n + 1);
  for (std::int64_t v = rp[0]; v < rp[1]; ++v)
  {
    const std::int64_t p = v % sqxy;
    x.insert(x.end(), {static_cast<double>(p % (n + 1)) / n,
                       static_cast<double>(p / (n + 1)) / n,
                       static_cast<double>(v / sqxy) / n});
  }

  return {std::move(cells), std::move(x)};
}
} // namespace

TEST_CASE("Geometric cell partitioner", "[geometric_partitioner]")
{
  MPI_Comm comm = MPI_COMM_WORLD;
  constexpr std::int64_t n = 12;
  auto [cells, x] = cube(comm, n);
  std::array<std::size_t, 2> xshape = {x.size() / 3, 3};
  fem::CoordinateElement<double> element(mesh::CellType::tetrahedron, 1);

  // Global entity counts for each entity dimension
  auto counts
      = [&cells, &x, xshape, &element, comm](
            const graph::AnyPartitionFunction& part, mesh::GhostMode ghost_mode)
  {
    mesh::Mesh<double> mesh = mesh::create_mesh(
        comm, comm,
        std::vector<std::span<const std::int64_t>>{
            std::span<const std::int64_t>(cells)},
        std::vector<fem::CoordinateElement<double>>{element}, comm, x, xshape,
        graph::Partitioner{.fn = part}, ghost_mode, 2, 1);
    mesh.topology_mutable()->create_entities(1);
    mesh.topology_mutable()->create_entities(2);

    std::array<std::int64_t, 5> c;
    for (int d = 0; d < 4; ++d)
      c[d] = mesh.topology()->index_map(d)->size_global();

    // Largest number of owned cells on any rank
    std::int64_t num_local = mesh.topology()->index_map(3)->size_local();
    MPI_Allreduce(&num_local, c.data() + 4, 1, MPI_INT64_T, MPI_MAX, comm);
    return c;
  };

  // graph::geom_partition_fn has no cell topology, so it never ghosts
  // regardless of the requested mode; compare against the unghosted
  // baseline.
  std::array<std::int64_t, 5> c0n
      = counts(graph::partition_graph, mesh::GhostMode::none);
  std::array<std::int64_t, 5> c1
      = counts(graph::partition_sfc_hilbert, mesh::GhostMode::none);

  CHECK(c0n[0] == (n + 1) * (n + 1) * (n + 1));
  CHECK(c0n[3] == 6 * n * n * n);
  for (int d = 0; d < 4; ++d)
    CHECK(c1[d] == c0n[d]);

  // Cells are shared out evenly (the partition is by equal count)
  const int size = dolfinx::MPI::size(comm);
  CHECK(c1[4] * size <= 1.1 * c1[3] + size);

#ifdef HAS_PARMETIS
  std::array<std::int64_t, 5> c3
      = counts(graph::parmetis::geom_partitioner, mesh::GhostMode::none);
  for (int d = 0; d < 4; ++d)
    CHECK(c3[d] == c0n[d]);
#endif

  for (mesh::GhostMode gm :
       {mesh::GhostMode::none, mesh::GhostMode::shared_facet})
  {
    std::array<std::int64_t, 5> c0 = counts(graph::partition_graph, gm);

    // Global entity counts do not depend on the partitioner
    CHECK(c0[0] == (n + 1) * (n + 1) * (n + 1));
    CHECK(c0[3] == 6 * n * n * n);

#ifdef HAS_PARMETIS
    std::array<std::int64_t, 5> c2
        = counts(graph::parmetis::geom_partitioner_kway(), gm);
    for (int d = 0; d < 4; ++d)
      CHECK(c2[d] == c0[d]);
#endif
  }
}

TEST_CASE("SFC point partition", "[partition_sfc]")
{
  MPI_Comm comm = MPI_COMM_WORLD;
  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);

  // Points on a line, distributed in blocks
  constexpr int num_points = 100;
  std::vector<double> x(num_points);
  for (int i = 0; i < num_points; ++i)
    x[i] = (num_points * rank + i) / static_cast<double>(num_points * size);

  for (auto partition :
       {&graph::partition_sfc_morton, &graph::partition_sfc_hilbert})
  {
    graph::AdjacencyList<std::int32_t> dest = graph::regular_adjacency_list(
        partition(comm, size, x, 1, std::nullopt), 1);
    REQUIRE(dest.num_nodes() == num_points);

    // There is exactly one destination per point (no graph, so no
    // ghosting is possible)
    CHECK(dest.array().size() == std::size_t(num_points));
    std::span<const std::int32_t> part = dest.array();
    CHECK(std::ranges::all_of(part,
                              [size](int p) { return p >= 0 and p < size; }));

    // Partition sizes are (approximately) equal
    std::vector<std::int64_t> count(size, 0), total(size, 0);
    for (int p : part)
      ++count[p];
    MPI_Allreduce(count.data(), total.data(), size, MPI_INT64_T, MPI_SUM, comm);
    const std::int64_t mx = *std::ranges::max_element(total);
    CHECK(mx <= 1.2 * num_points + 1);

    // Points are assigned in space-filling-curve order, which in one
    // dimension is sorted order for both curves
    CHECK(std::ranges::is_sorted(part));

    // Unit weights must take the same path through the weighted sampler
    // without changing the result.
    std::vector<std::int32_t> weights(num_points, 1);
    std::vector<int> weighted_part
        = partition(comm, size, x, 1, std::span<const std::int32_t>(weights));
    CHECK(weighted_part == std::vector<int>(part.begin(), part.end()));

    for (int i = 0; i < num_points; ++i)
      weights[i] = 1 + i % 5;
    weighted_part
        = partition(comm, size, x, 1, std::span<const std::int32_t>(weights));
    CHECK(std::ranges::is_sorted(weighted_part));
    std::vector<std::int64_t> local_weight(size, 0), total_weight(size, 0);
    for (int i = 0; i < num_points; ++i)
      local_weight[weighted_part[i]] += weights[i];
    MPI_Allreduce(local_weight.data(), total_weight.data(), size, MPI_INT64_T,
                  MPI_SUM, comm);
    CHECK(*std::ranges::max_element(total_weight)
          <= 1.2
                     * std::accumulate(total_weight.begin(), total_weight.end(),
                                       std::int64_t(0))
                     / size
                 + 5);

    // A single partition holds everything
    graph::AdjacencyList<std::int32_t> part1 = graph::regular_adjacency_list(
        partition(comm, 1, x, 1, std::nullopt), 1);
    CHECK(std::ranges::all_of(part1.array(), [](int p) { return p == 0; }));
  }
}

TEST_CASE("Geometric partitioning with distributed geometry",
          "[geometric_partitioner]")
{
  MPI_Comm comm = MPI_COMM_WORLD;
  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);
  const std::int64_t n = std::max<std::int64_t>(8, 2 * size);

  std::vector<std::int64_t> cells;
  MPI_Comm commt = MPI_COMM_NULL;
  if (rank == 0)
  {
    commt = MPI_COMM_SELF;
    cells.resize(2 * n);
    for (std::int64_t i = 0; i < n; ++i)
    {
      cells[2 * i] = i;
      cells[2 * i + 1] = i + 1;
    }
  }

  const auto range = common::local_range(rank, n + 1, size);
  std::vector<double> x(range[1] - range[0]);
  for (std::int64_t i = range[0]; i < range[1]; ++i)
    x[i - range[0]] = static_cast<double>(i) / n;

  fem::CoordinateElement<double> element(mesh::CellType::interval, 1);
  auto check_partitioner = [&](const graph::AnyPartitionFunction& partitioner)
  {
    mesh::Mesh<double> msh = mesh::create_mesh(
        comm, commt, std::span<const std::int64_t>(cells), element, comm, x,
        {x.size(), 1}, graph::Partitioner{.fn = partitioner},
        mesh::GhostMode::none, 2, 1);
    CHECK(msh.topology()->index_map(1)->size_global() == n);
  };

  check_partitioner(graph::partition_sfc_hilbert);
#ifdef HAS_PARMETIS
  check_partitioner(graph::parmetis::geom_partitioner);
#endif
}

TEST_CASE("SFC curve properties", "[partition_sfc]")
{
  // One part per point makes a point's part index its curve position,
  // so traversal order can be checked directly. MPI_COMM_SELF lets
  // every rank test the curve over the whole grid.
  constexpr int k = 8;
  std::vector<double> x;
  std::vector<std::array<int, 3>> grid;
  for (int i = 0; i < k; ++i)
  {
    for (int j = 0; j < k; ++j)
    {
      for (int l = 0; l < k; ++l)
      {
        x.insert(x.end(), {double(i), double(j), double(l)});
        grid.push_back({i, j, l});
      }
    }
  }
  const int num_points = grid.size();

  auto curve_order = [&x, num_points](auto partition)
  {
    graph::AdjacencyList<std::int32_t> dest = graph::regular_adjacency_list(
        partition(MPI_COMM_SELF, num_points, x, 3, std::nullopt), 1);
    std::span<const std::int32_t> part = dest.array();

    // The part indices must be a permutation of [0, num_points), i.e.
    // the curve visits every point exactly once
    std::vector<int> sorted(part.begin(), part.end());
    std::ranges::sort(sorted);
    for (int i = 0; i < num_points; ++i)
      REQUIRE(sorted[i] == i);

    std::vector<int> order(num_points);
    for (int i = 0; i < num_points; ++i)
      order[part[i]] = i;
    return order;
  };

  // Largest step in grid units between successive points on the curve
  auto max_step = [&grid](const std::vector<int>& order)
  {
    int step = 0;
    for (std::size_t p = 0; p + 1 < order.size(); ++p)
    {
      std::array<int, 3> a = grid[order[p]], b = grid[order[p + 1]];
      step = std::max(step, std::abs(a[0] - b[0]) + std::abs(a[1] - b[1])
                                + std::abs(a[2] - b[2]));
    }
    return step;
  };

  // Successive points on a Hilbert curve are always neighbours in space.
  // A Morton curve, in contrast, jumps.
  CHECK(max_step(curve_order(&graph::partition_sfc_hilbert)) == 1);
  CHECK(max_step(curve_order(&graph::partition_sfc_morton)) > 1);
}

TEST_CASE("Geometric cell reordering", "[geometric_partitioner]")
{
  const std::vector<std::int64_t> cells = {0, 1, 2, 1, 3, 2};
  const std::vector<double> x = {0, 0, 1, 0, 0, 1, 1, 1};
  const fem::CoordinateElement<double> element(mesh::CellType::triangle, 1);

  const graph::Reorder reorder = graph::reorder_geom_fn(
      [](std::span<const double> centroids, int gdim)
      {
        REQUIRE(gdim == 2);
        REQUIRE(centroids.size() == 4);
        return std::vector<std::int32_t>{1, 0};
      });
  mesh::Mesh<double> msh = mesh::create_mesh(
      MPI_COMM_SELF, MPI_COMM_SELF, std::span<const std::int64_t>(cells),
      element, MPI_COMM_SELF, x, {4, 2}, graph::Partitioner{},
      mesh::GhostMode::none, 2, 1, reorder);

  CHECK(msh.topology()->original_cell_index[0]
        == std::vector<std::int64_t>{1, 0});
}

TEST_CASE("Default cell reordering", "[geometric_partitioner]")
{
  const std::vector<std::int64_t> cells = {0, 1, 2, 1, 3, 2};
  const std::vector<double> x = {0, 0, 1, 0, 0, 1, 1, 1};
  const fem::CoordinateElement<double> element(mesh::CellType::triangle, 1);

  mesh::Mesh<double> msh_default = mesh::create_mesh(
      MPI_COMM_SELF, MPI_COMM_SELF, std::span<const std::int64_t>(cells),
      element, MPI_COMM_SELF, x, {4, 2}, graph::Partitioner{},
      mesh::GhostMode::none, 2, 1);
  const graph::Reorder reorder;
  mesh::Mesh<double> msh_empty = mesh::create_mesh(
      MPI_COMM_SELF, MPI_COMM_SELF, std::span<const std::int64_t>(cells),
      element, MPI_COMM_SELF, x, {4, 2}, graph::Partitioner{},
      mesh::GhostMode::none, 2, 1, reorder);

  CHECK(msh_empty.topology()->original_cell_index
        == msh_default.topology()->original_cell_index);
}

TEST_CASE("SFC point reordering", "[partition_sfc]")
{
  const std::vector<double> x = {3, 1, 2, 0};
  for (auto reorder : {graph::reorder_sfc_morton, graph::reorder_sfc_hilbert})
  {
    const std::vector<std::int32_t> map = reorder(x, 1);
    std::vector<double> x_ordered(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      x_ordered[map[i]] = x[i];
    CHECK(std::ranges::is_sorted(x_ordered));
  }
}
