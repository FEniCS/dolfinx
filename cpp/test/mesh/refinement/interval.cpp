// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <optional>
#include <span>

#include <mpi.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/interval.h>
#include <dolfinx/refinement/refine.h>

#include "../util.h"

using namespace dolfinx;
using namespace Catch::Matchers;

template <typename T>
mesh::Mesh<T> create_3_vertex_interval_mesh()
{
  // creates mesh with vertices
  std::array<T, 3> v0 = {0., 0., 0.};
  std::array<T, 3> v1 = {.5, 1., 2.};
  std::array<T, 3> v2 = {1., 2., 4.};

  // and connectivity
  // v0 --- v1 --- v2
  std::vector<std::int64_t> cells{0, 1, 1, 2};

  std::vector<T> x{v0[0], v0[1], v0[2], v1[0], v1[1],
                   v1[2], v2[0], v2[1], v2[2]};
  fem::CoordinateElement<T> element(mesh::CellType::interval, 1);
  return mesh::create_mesh(MPI_COMM_SELF, MPI_COMM_SELF, cells, element,
                           MPI_COMM_SELF, x, {x.size() / 3, 3},
                           mesh::create_cell_partitioner());
}

TEMPLATE_TEST_CASE("Interval uniform refinement",
                   "[refinement][interval][uniform]", float, double)
{
  using T = TestType;

  mesh::Mesh<T> mesh = create_3_vertex_interval_mesh<T>();
  mesh.topology()->create_connectivity(1, 0);

  // TODO: parent_facet
  auto [refined_mesh, parent_edge, parent_facet] = refinement::refine(
      mesh, std::nullopt, false, mesh::GhostMode::shared_facet,
      refinement::Option::parent_cell);

  std::vector<T> expected_x = {
      /* v_0 */ 0.0, 0.0, 0.0,
      /* v_1 */ .25, 0.5, 1.0,
      /* v_2 */ 0.5, 1.0, 2.0,
      /* v_3 */ .75, 1.5, 3.0,
      /* v_4 */ 1.0, 2.0, 4.0,
  };

  CHECK_THAT(refined_mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  // Check topology
  auto topology = refined_mesh.topology_mutable();
  CHECK(topology->dim() == 1);

  topology->create_connectivity(0, 1);
  CHECK_adjacency_list_equal(*topology->connectivity(0, 1), {/* v_0 */ {0},
                                                             /* v_1 */ {0, 1},
                                                             /* v_2 */ {1, 2},
                                                             /* v_3 */ {2, 3},
                                                             /* v_4 */ {3}});

  CHECK_THAT(parent_edge.value(),
             RangeEquals(std::vector<std::int32_t>{0, 0, 1, 1}));
}

TEMPLATE_TEST_CASE("Interval adaptive refinement",
                   "[refinement][interval][adaptive]", float, double)
{
  using T = TestType;

  mesh::Mesh<T> mesh = create_3_vertex_interval_mesh<T>();
  mesh.topology()->create_connectivity(1, 0);

  std::vector<std::int32_t> edges{1};
  // TODO: parent_facet
  auto [refined_mesh, parent_edge, parent_facet] = refinement::refine(
      mesh, std::span(edges), false, mesh::GhostMode::shared_facet,
      refinement::Option::parent_cell);

  std::vector<T> expected_x = {
      /* v_0 */ 0.0, 0.0, 0.0,
      /* v_1 */ 0.5, 1.0, 2.0,
      /* v_2 */ .75, 1.5, 3.0,
      /* v_3 */ 1.0, 2.0, 4.0,
  };

  CHECK_THAT(refined_mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  auto topology = refined_mesh.topology_mutable();
  CHECK(topology->dim() == 1);

  topology->create_connectivity(0, 1);
  CHECK_adjacency_list_equal(*topology->connectivity(0, 1), {/* v_0 */ {0},
                                                             /* v_1 */ {0, 1},
                                                             /* v_2 */ {1, 2},
                                                             /* v_3 */ {2}});

  CHECK_THAT(parent_edge.value(),
             RangeEquals(std::vector<std::int32_t>{0, 1, 1}));
}

TEMPLATE_TEST_CASE("Interval Refinement (parallel)",
                   "[refinement][interva][parallel]", float, double)
{
  /**
  Produces an interval with communicator size intervals. Every process is
  assigned one interval and we refine uniformly.
  */

  using T = TestType;

  const int comm_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

  if (comm_size == 1)
    SKIP("Only runs in parallel");

  auto create_mesh = [&]()
  {
    std::vector<T> x;
    std::vector<std::int64_t> cells;
    fem::CoordinateElement<T> element(mesh::CellType::interval, 1);
    if (rank == 0)
    {
      for (std::int64_t i = 0; i < comm_size + 1; i++)
        x.insert(x.end(), {static_cast<T>(i) / comm_size, static_cast<T>(i) + 1,
                           static_cast<T>(2. * i + comm_size)});
      for (std::int64_t i = 0; i < 2 * comm_size; i++)
      {
        auto div = std::div(i, static_cast<std::int64_t>(2));
        cells.push_back(div.quot + div.rem);
      }
    }

    auto partitioner
        = [](MPI_Comm /* comm */, int /* nparts */,
             const std::vector<mesh::CellType>& /* cell_types */,
             const std::vector<std::span<const std::int64_t>>& /* cells */)
        -> graph::AdjacencyList<std::int32_t>
    {
      return graph::AdjacencyList<std::int32_t>(
          dolfinx::MPI::size(MPI_COMM_WORLD));
    };

    MPI_Comm commt = rank == 0 ? MPI_COMM_SELF : MPI_COMM_NULL;
    return mesh::create_mesh(MPI_COMM_WORLD, commt, cells, element, commt, x,
                             {x.size() / 3, 3}, partitioner);
  };

  mesh::Mesh<T> mesh = create_mesh();
  mesh.topology()->create_connectivity(1, 0);

  // TODO: parent_facet
  auto [refined_mesh, parent_edges, parent_facet] = refinement::refine(
      mesh, std::nullopt, false, mesh::GhostMode::shared_facet,
      refinement::Option::parent_cell);

  T rank_d = static_cast<T>(rank);
  T comm_size_d = static_cast<T>(comm_size);

  auto x = refined_mesh.geometry().x();
  std::ranges::sort(x);
  std::vector<T> expected_x
      = {rank_d / comm_size_d,
         static_cast<T>(rank_d / comm_size_d + (1. / (2 * comm_size_d))),
         static_cast<T>(rank_d / comm_size_d + (2. / (2 * comm_size_d))),
         rank_d + 1,
         static_cast<T>(rank_d + 1.5),
         rank_d + 2,
         2 * rank_d + comm_size_d,
         static_cast<T>(2 * (rank_d + .5) + comm_size_d),
         2 * (rank_d + 1) + comm_size_d};
  CHECK_THAT(x, RangeEquals(expected_x, [](auto a, auto b)
                            { return std::abs(a - b) <= EPS<T>; }));

  // Check topology
  {
    auto topology = refined_mesh.topology_mutable();
    CHECK(topology->dim() == 1);

    topology->create_connectivity(0, 1);
    auto v_to_e = topology->connectivity(0, 1);

    // find the center index, i.e. the one with two outgoing edges
    std::size_t center_index = v_to_e->num_links(0) == 2   ? 0
                               : v_to_e->num_links(1) == 2 ? 1
                                                           : 2;
    CHECK(v_to_e->num_links(center_index) == 2);
    // check it's connected to both edge 0 and 1
    CHECK(std::ranges::find(v_to_e->links(center_index), 0)
          != v_to_e->links(center_index).end());
    CHECK(std::ranges::find(v_to_e->links(center_index), 1)
          != v_to_e->links(center_index).end());

    // side vertices are only connected to one edge
    CHECK(v_to_e->links((center_index + 1) % 3).size() == 1);
    CHECK(v_to_e->links((center_index + 2) % 3).size() == 1);

    // and this edge is not shared
    CHECK(v_to_e->links((center_index + 1) % 3)[0]
          != v_to_e->links((center_index + 2) % 3)[0]);
  }
}
