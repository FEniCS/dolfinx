// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/matchers/catch_matchers.hpp>
#include <limits>
#include <optional>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/interval.h>
#include <dolfinx/refinement/refine.h>
#include <span>

using namespace dolfinx;
using namespace Catch::Matchers;

constexpr auto EPS = std::numeric_limits<double>::epsilon();

mesh::Mesh<double> create_3_vertex_interval_mesh()
{
  // creates mesh with vertices
  std::array<double, 3> v0 = {0., 0., 0.};
  std::array<double, 3> v1 = {.5, 1., 2.};
  std::array<double, 3> v2 = {1., 2., 4.};

  // and connectivity
  // v0 --- v1 --- v2
  std::vector<std::int64_t> cells{0, 1, 1, 2};

  std::vector<double> x{v0[0], v0[1], v0[2], v1[0], v1[1],
                        v1[2], v2[0], v2[1], v2[2]};
  fem::CoordinateElement<double> element(mesh::CellType::interval, 1);
  return mesh::create_mesh(MPI_COMM_SELF, MPI_COMM_SELF, cells, element,
                           MPI_COMM_SELF, x, {x.size() / 3, 3},
                           mesh::create_cell_partitioner());
}

TEST_CASE("Interval uniform refinement", "refinement,interval,uniform")
{
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) > 1)
    return;

  mesh::Mesh<double> mesh = create_3_vertex_interval_mesh();
  mesh.topology()->create_connectivity(1, 0);

  auto [refined_mesh, parent_edge]
      = refinement::refine_interval(mesh, std::nullopt, false);

  // Check geometry
  {
    const auto& x = refined_mesh.geometry().x();

    CHECK(x.size() == 15);

    REQUIRE_THAT(x[0], WithinAbs(0.0, EPS));
    REQUIRE_THAT(x[1], WithinAbs(0.0, EPS));
    REQUIRE_THAT(x[2], WithinAbs(0.0, EPS));

    REQUIRE_THAT(x[3], WithinAbs(0.25, EPS));
    REQUIRE_THAT(x[4], WithinAbs(0.5, EPS));
    REQUIRE_THAT(x[5], WithinAbs(1.0, EPS));

    REQUIRE_THAT(x[6], WithinAbs(0.5, EPS));
    REQUIRE_THAT(x[7], WithinAbs(1.0, EPS));
    REQUIRE_THAT(x[8], WithinAbs(2.0, EPS));

    REQUIRE_THAT(x[9], WithinAbs(0.75, EPS));
    REQUIRE_THAT(x[10], WithinAbs(1.5, EPS));
    REQUIRE_THAT(x[11], WithinAbs(3.0, EPS));

    REQUIRE_THAT(x[12], WithinAbs(1.0, EPS));
    REQUIRE_THAT(x[13], WithinAbs(2.0, EPS));
    REQUIRE_THAT(x[14], WithinAbs(4.0, EPS));
  }

  // Check topology
  {
    auto topology = refined_mesh.topology_mutable();
    CHECK(topology->dim() == 1);

    topology->create_connectivity(0, 1);
    auto v_to_e = topology->connectivity(0, 1);

    CHECK(v_to_e->num_links(0) == 1);
    CHECK(v_to_e->links(0)[0] == 0);

    CHECK(v_to_e->num_links(1) == 2);
    CHECK(v_to_e->links(1)[0] == 0);
    CHECK(v_to_e->links(1)[1] == 1);

    CHECK(v_to_e->num_links(2) == 2);
    CHECK(v_to_e->links(2)[0] == 1);
    CHECK(v_to_e->links(2)[1] == 2);

    CHECK(v_to_e->num_links(3) == 2);
    CHECK(v_to_e->links(3)[0] == 2);
    CHECK(v_to_e->links(3)[1] == 3);

    CHECK(v_to_e->num_links(4) == 1);
    CHECK(v_to_e->links(4)[0] == 3);
  }

  // Check parent edges
  {
    CHECK(parent_edge.size() == 4);

    CHECK(parent_edge[0] == 0);
    CHECK(parent_edge[1] == 0);
    CHECK(parent_edge[2] == 1);
    CHECK(parent_edge[3] == 1);
  }
}

TEST_CASE("Interval adaptive refinement", "refinement,interval,adaptive")
{
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) > 1)
    return;

  mesh::Mesh<double> mesh = create_3_vertex_interval_mesh();
  mesh.topology()->create_connectivity(1, 0);

  std::vector<std::int32_t> edges{1};
  auto [refined_mesh, parent_edge]
      = refinement::refine_interval(mesh, std::span(edges), false);

  // Check geometry
  {
    const auto& x = refined_mesh.geometry().x();

    CHECK(x.size() == 12); // -> 5 vertices
    REQUIRE_THAT(x[0], WithinAbs(0.0, EPS));
    REQUIRE_THAT(x[1], WithinAbs(0.0, EPS));
    REQUIRE_THAT(x[2], WithinAbs(0.0, EPS));

    REQUIRE_THAT(x[3], WithinAbs(0.5, EPS));
    REQUIRE_THAT(x[4], WithinAbs(1.0, EPS));
    REQUIRE_THAT(x[5], WithinAbs(2.0, EPS));

    REQUIRE_THAT(x[6], WithinAbs(0.75, EPS));
    REQUIRE_THAT(x[7], WithinAbs(1.5, EPS));
    REQUIRE_THAT(x[8], WithinAbs(3.0, EPS));

    REQUIRE_THAT(x[9], WithinAbs(1.0, EPS));
    REQUIRE_THAT(x[10], WithinAbs(2.0, EPS));
    REQUIRE_THAT(x[11], WithinAbs(4.0, EPS));
  }

  // Check topology
  {
    auto topology = refined_mesh.topology_mutable();
    CHECK(topology->dim() == 1);

    topology->create_connectivity(0, 1);
    auto v_to_e = topology->connectivity(0, 1);

    CHECK(v_to_e->num_links(0) == 1);
    CHECK(v_to_e->links(0)[0] == 0);

    CHECK(v_to_e->num_links(1) == 2);
    CHECK(v_to_e->links(1)[0] == 0);
    CHECK(v_to_e->links(1)[1] == 1);

    CHECK(v_to_e->num_links(2) == 2);
    CHECK(v_to_e->links(2)[0] == 1);
    CHECK(v_to_e->links(2)[1] == 2);

    CHECK(v_to_e->num_links(3) == 1);
    CHECK(v_to_e->links(3)[0] == 2);
  }

  // Check parent edges
  {
    CHECK(parent_edge.size() == 3);

    CHECK(parent_edge[0] == 0);
    CHECK(parent_edge[1] == 1);
    CHECK(parent_edge[2] == 1);
  }
}

TEST_CASE("Interval Refinement (parallel)", "refinement,interval,paralle")
{
  /**
  Produces an interval with communicator size intervals. Every process is
  assigned one interval and we refine uniformly.
  */

  const auto comm_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const auto rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

  if (comm_size == 1)
    return;

  auto create_mesh = [&]()
  {
    std::vector<double> x;
    std::vector<std::int64_t> cells;
    fem::CoordinateElement<double> element(mesh::CellType::interval, 1);
    if (rank == 0)
    {
      for (std::int64_t i = 0; i < comm_size + 1; i++)
        x.insert(x.end(), {static_cast<double>(i) / comm_size,
                           static_cast<double>(i) + 1, 2. * i + comm_size});
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

    auto commt = rank == 0 ? MPI_COMM_SELF : MPI_COMM_NULL;
    return mesh::create_mesh(MPI_COMM_WORLD, commt, cells, element, commt, x,
                             {x.size() / 3, 3}, partitioner);
  };

  mesh::Mesh<double> mesh = create_mesh();
  mesh.topology()->create_connectivity(1, 0);

  // complete refinement
  {
    auto [refined_mesh, parent_edges]
        = refinement::refine_interval(mesh, std::nullopt, false);

    // Check geometry
    {

      auto x = refined_mesh.geometry().x();
      CHECK(x.size() == 9);

      std::ranges::sort(x);

      double rank_d = static_cast<double>(rank);
      double comm_size_d = static_cast<double>(comm_size);

      REQUIRE_THAT(x[0], WithinAbs(rank_d / comm_size_d, EPS));
      REQUIRE_THAT(
          x[1], WithinAbs(rank_d / comm_size_d + 1. / (2 * comm_size_d), EPS));
      REQUIRE_THAT(
          x[2], WithinAbs(rank_d / comm_size_d + 2. / (2 * comm_size_d), EPS));

      REQUIRE_THAT(x[3], WithinAbs(rank_d + 1, EPS));
      REQUIRE_THAT(x[4], WithinAbs(rank_d + 1.5, EPS));
      REQUIRE_THAT(x[5], WithinAbs(rank_d + 2, EPS));

      REQUIRE_THAT(x[6], WithinAbs(2 * rank_d + comm_size_d, EPS));
      REQUIRE_THAT(x[7], WithinAbs(2 * (rank_d + .5) + comm_size_d, EPS));
      REQUIRE_THAT(x[8], WithinAbs(2 * (rank_d + 1) + comm_size_d, EPS));
    }

    // Check topology
    {
      auto topology = refined_mesh.topology_mutable();
      CHECK(topology->dim() == 1);

      topology->create_connectivity(0, 1);
      auto v_to_e = topology->connectivity(0, 1);

      // find the center index, i.e. the one with two outgoing edges
      int center_index = v_to_e->num_links(0) == 2   ? 0
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
}
