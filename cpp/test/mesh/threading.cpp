// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for threaded mesh creation. Mesh construction and entity
// computation are split across threads by range, with each range
// writing its own output and the ranges joined in order, so a threaded
// run must reproduce a serial one exactly -- not merely agree on
// entity counts.
//
// Note: run on more than one rank for full cover. The dual graph only
// reaches the result through the partitioner, and entity sharing only
// exists between ranks, so a single-rank run exercises the threading in
// graphbuild.cpp and topologycomputation.cpp without being able to
// observe it.

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/local_range.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <span>
#include <utility>
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

/// @brief Everything about a mesh's topology that threading must not
/// perturb: for each dimension, the index map (local/ghost counts,
/// ghost global indices and their owners) and the entity-vertex
/// connectivity, plus the cell permutations and inter-process facets.
struct Fingerprint
{
  std::vector<std::int64_t> ints;
  std::vector<std::uint32_t> perms;

  bool operator==(const Fingerprint&) const = default;
};

/// @brief Build the cube mesh with `num_threads` threads and fingerprint
/// its topology.
Fingerprint build(MPI_Comm comm, std::span<const std::int64_t> cells,
                  std::span<const double> x, std::array<std::size_t, 2> xshape,
                  int num_threads)
{
  fem::CoordinateElement<double> element(mesh::CellType::tetrahedron, 1);
  mesh::Mesh<double> mesh = mesh::create_mesh(
      comm, comm, std::vector<std::span<const std::int64_t>>{cells},
      std::vector<fem::CoordinateElement<double>>{element}, comm, x, xshape,
      graph::Partitioner{.fn = graph::partition_graph},
      mesh::GhostMode::shared_facet, 2, num_threads);

  const int tdim = mesh.topology()->dim();
  for (int d = 1; d < tdim; ++d)
    mesh.topology_mutable()->create_entities(d, num_threads);
  mesh.topology_mutable()->create_cell_permutations(num_threads);

  Fingerprint f;
  for (int d = 0; d <= tdim; ++d)
  {
    auto im = mesh.topology()->index_map(d);
    REQUIRE(im);
    f.ints.insert(f.ints.end(), {im->size_local(), im->num_ghosts()});
    f.ints.insert(f.ints.end(), im->ghosts().begin(), im->ghosts().end());
    f.ints.insert(f.ints.end(), im->owners().begin(), im->owners().end());

    if (auto c = mesh.topology()->connectivity(d, 0))
    {
      f.ints.insert(f.ints.end(), c->array().begin(), c->array().end());
      f.ints.insert(f.ints.end(), c->offsets().begin(), c->offsets().end());
    }
  }

  const std::vector<std::int32_t>& ipf = mesh.topology()->interprocess_facets();
  f.ints.insert(f.ints.end(), ipf.begin(), ipf.end());

  f.perms = mesh.topology()->get_cell_permutation_info();
  return f;
}
} // namespace

TEST_CASE("Threaded mesh creation matches serial", "[mesh][threading]")
{
  MPI_Comm comm = MPI_COMM_WORLD;
  constexpr std::int64_t n = 12;
  auto [cells, x] = cube(comm, n);
  std::array<std::size_t, 2> xshape = {x.size() / 3, 3};

  const Fingerprint serial = build(comm, cells, x, xshape, 1);

  // More threads than the hardware has is deliberate: the work split is
  // by range, not by core, so the results must not depend on either
  for (int num_threads : {2, 3, 4, 8, 16})
  {
    INFO("num_threads = " << num_threads);
    CHECK(build(comm, cells, x, xshape, num_threads) == serial);
  }
}

TEST_CASE("Mesh creation rejects num_threads < 1", "[mesh][threading]")
{
  MPI_Comm comm = MPI_COMM_WORLD;
  constexpr std::int64_t n = 2;
  auto [cells, x] = cube(comm, n);
  std::array<std::size_t, 2> xshape = {x.size() / 3, 3};
  CHECK_THROWS_AS(build(comm, cells, x, xshape, 0), std::invalid_argument);
}
