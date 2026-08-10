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
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
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
  auto counts = [&cells, &x, xshape, &element,
                 comm](const mesh::CellPartitionFunction& part)
  {
    mesh::Mesh<double> mesh = mesh::create_mesh(
        comm, comm,
        std::vector<std::span<const std::int64_t>>{
            std::span<const std::int64_t>(cells)},
        std::vector<fem::CoordinateElement<double>>{element}, comm, x, xshape,
        part, 2, 1);
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

  for (mesh::GhostMode gm :
       {mesh::GhostMode::none, mesh::GhostMode::shared_facet})
  {
    std::array<std::int64_t, 5> c0
        = counts(mesh::create_cell_partitioner(gm, 2));
    std::array<std::int64_t, 5> c1
        = counts(mesh::create_geometric_cell_partitioner<double>(
            gm, comm, std::span<const double>(x), xshape, 2));

    // Global entity counts do not depend on the partitioner
    CHECK(c0[0] == (n + 1) * (n + 1) * (n + 1));
    CHECK(c0[3] == 6 * n * n * n);
    for (int d = 0; d < 4; ++d)
      CHECK(c1[d] == c0[d]);

    // Cells are shared out evenly (the partition is by equal count)
    const int size = dolfinx::MPI::size(comm);
    CHECK(c1[4] * size <= 1.1 * c1[3] + size);

#ifdef HAS_PARMETIS
    for (graph::parmetis::geom_method method :
         {graph::parmetis::geom_method::kway,
          graph::parmetis::geom_method::curve})
    {
      std::array<std::int64_t, 5> c2
          = counts(mesh::create_geometric_cell_partitioner<double>(
              gm, comm, std::span<const double>(x), xshape, 2, 1,
              graph::parmetis::geom_partitioner(method)));
      for (int d = 0; d < 4; ++d)
        CHECK(c2[d] == c0[d]);
    }
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

  std::vector<int> part = graph::partition_sfc(comm, size, x, 1);
  REQUIRE(part.size() == num_points);
  CHECK(
      std::ranges::all_of(part, [size](int p) { return p >= 0 and p < size; }));

  // Partition sizes are (approximately) equal
  std::vector<std::int64_t> count(size, 0), total(size, 0);
  for (int p : part)
    ++count[p];
  MPI_Allreduce(count.data(), total.data(), size, MPI_INT64_T, MPI_SUM, comm);
  const std::int64_t mx = *std::ranges::max_element(total);
  CHECK(mx <= 1.2 * num_points + 1);

  // Points are assigned in space-filling-curve (here, sorted) order
  CHECK(std::ranges::is_sorted(part));

  // A single partition holds everything
  std::vector<int> part1 = graph::partition_sfc(comm, 1, x, 1);
  CHECK(std::ranges::all_of(part1, [](int p) { return p == 0; }));
}
