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

#include "cube.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

namespace
{

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
                  int num_threads,
                  const graph::partition_fn& partitioner
                  = graph::partition_graph)
{
  fem::CoordinateElement<double> element(mesh::CellType::tetrahedron, 1);
  mesh::Mesh<double> mesh = mesh::create_mesh(
      comm, comm, std::vector<std::span<const std::int64_t>>{cells},
      std::vector<fem::CoordinateElement<double>>{element}, comm, x, xshape,
      graph::Partitioner{.fn = partitioner}, mesh::GhostMode::shared_facet, 2,
      num_threads);

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
  auto [cells, x] = test::cube(comm, n);
  std::array<std::size_t, 2> xshape = {x.size() / 3, 3};

  // Partition once: repeated partitioner calls can produce different
  // destinations for the same graph. Compare graphs before replaying.
  std::optional<graph::AdjacencyList<std::int64_t>> serial_graph;
  std::optional<graph::AdjacencyList<std::int32_t>> destinations;
  const graph::partition_fn partitioner
      = [&serial_graph, &destinations](
            MPI_Comm c, int nparts, const graph::AdjacencyList<std::int64_t>& g,
            std::optional<std::span<const std::int32_t>> node_weights,
            std::optional<std::span<const std::int32_t>> edge_weights,
            bool ghosting) -> graph::AdjacencyList<std::int32_t>
  {
    if (!serial_graph)
    {
      serial_graph = g;
      destinations = graph::partition_graph(c, nparts, g, node_weights,
                                            edge_weights, ghosting);
    }
    else
      CHECK(g == *serial_graph);
    return *destinations;
  };

  const Fingerprint serial = build(comm, cells, x, xshape, 1, partitioner);

  // More threads than the hardware has is deliberate: the work split is
  // by range, not by core, so the results must not depend on either
  for (int num_threads : {2, 3, 4, 8, 16})
  {
    INFO("num_threads = " << num_threads);
    CHECK(build(comm, cells, x, xshape, num_threads, partitioner) == serial);
  }
}

TEST_CASE("Mesh creation rejects num_threads < 1", "[mesh][threading]")
{
  MPI_Comm comm = MPI_COMM_WORLD;
  constexpr std::int64_t n = 2;
  auto [cells, x] = test::cube(comm, n);
  std::array<std::size_t, 2> xshape = {x.size() / 3, 3};
  CHECK_THROWS_AS(build(comm, cells, x, xshape, 0), std::invalid_argument);
}
