// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <optional>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>

#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/interval.h>
#include <dolfinx/refinement/refine.h>
#include <span>

using namespace dolfinx;

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

    CHECK(x[0] == 0.0);
    CHECK(x[1] == 0.0);
    CHECK(x[2] == 0.0);

    CHECK(x[3] == 0.25);
    CHECK(x[4] == 0.5);
    CHECK(x[5] == 1.0);

    CHECK(x[6] == 0.5);
    CHECK(x[7] == 1.0);
    CHECK(x[8] == 2.0);

    CHECK(x[9] == 0.75);
    CHECK(x[10] == 1.5);
    CHECK(x[11] == 3.0);

    CHECK(x[12] == 1.0);
    CHECK(x[13] == 2.0);
    CHECK(x[14] == 4.0);
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
    CHECK(x[0] == 0.0);
    CHECK(x[1] == 0.0);
    CHECK(x[2] == 0.0);

    CHECK(x[3] == 0.5);
    CHECK(x[4] == 1.0);
    CHECK(x[5] == 2.0);

    CHECK(x[6] == 0.75);
    CHECK(x[7] == 1.5);
    CHECK(x[8] == 3.0);

    CHECK(x[9] == 1.0);
    CHECK(x[10] == 2.0);
    CHECK(x[11] == 4.0);
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
  assigned one intervall and we refine uniformly.
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

      CHECK(x[0] == static_cast<double>(rank) / comm_size);
      CHECK(x[1]
            == static_cast<double>(rank) / comm_size + 1. / (2 * comm_size));
      CHECK(x[2]
            == static_cast<double>(rank) / comm_size + 2. / (2 * comm_size));

      CHECK(x[3] == rank + 1);
      CHECK(x[4] == rank + 1.5);
      CHECK(x[5] == rank + 2);

      CHECK(x[6] == 2 * rank + comm_size);
      CHECK(x[7] == 2 * (rank + .5) + comm_size);
      CHECK(x[8] == 2 * (rank + 1) + comm_size);
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
