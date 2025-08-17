// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx/mesh/generation.h"

#include <algorithm>
#include <iterator>
#include <mpi.h>
#include <vector>

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

using namespace dolfinx;
using namespace Catch::Matchers;

namespace
{
template <typename T>
void CHECK_adjacency_list_equal(
    const dolfinx::graph::AdjacencyList<T>& adj_list,
    const std::vector<std::vector<T>>& expected_list)
{
  REQUIRE(static_cast<std::size_t>(adj_list.num_nodes())
          == expected_list.size());

  for (T i = 0; i < adj_list.num_nodes(); i++)
  {
    CHECK_THAT(adj_list.links(i),
               Catch::Matchers::RangeEquals(expected_list[i]));
  }
}

template <typename T>
constexpr auto EPS = std::numeric_limits<T>::epsilon();
} // namespace

TEMPLATE_TEST_CASE("Interval mesh", "[mesh][interval]", float, double)
{
  using T = TestType;
  mesh::Mesh<T> mesh = mesh::create_interval<T>(MPI_COMM_SELF, 4, {0.0, 1.0});

  {
    int comp_result;
    MPI_Comm_compare(mesh.comm(), MPI_COMM_SELF, &comp_result);
    CHECK(comp_result == MPI_CONGRUENT);
  }

  CHECK(mesh.geometry().dim() == 1);

  // vertex layout
  // 0 --- 1 --- 2 --- 3 --- 4
  std::vector<T> expected_x = {
      /* v_0 */ 0.0,
      /* v_1 */ 0.25,
      /* v_2 */ 0.5,
      /* v_3 */ 0.75,
      /* v_4 */ 1.0,
  };

  auto [x, cells] = mesh::impl::create_interval_cells<T>({0., 1.}, 4);

  CHECK(mesh.topology()->index_map(0)->size_local() == 5);
  for (std::int64_t i = 0; i < mesh.topology()->index_map(0)->size_local(); i++)
  {
    CHECK(std::abs(expected_x[mesh.geometry().input_global_indices()[i]]
                   - mesh.geometry().x()[3 * i])
          <= EPS<T>);
    CHECK(std::abs(mesh.geometry().x()[3 * i + 1] - 0.0) <= EPS<T>);
    CHECK(std::abs(mesh.geometry().x()[3 * i + 2] - 0.0) <= EPS<T>);
  }

  // cell layout
  // x -0- x -1- x -2- x -3- x
  mesh.topology()->create_connectivity(0, 1);
  CHECK_adjacency_list_equal(*mesh.topology()->connectivity(0, 1),
                             {{0}, {0, 1}, {1, 2}, {2, 3}, {3}});
}

TEMPLATE_TEST_CASE("Interval mesh (parallel)", "[mesh][interval]", float,
                   double)
{
  using T = TestType;
  int comm_size = dolfinx::MPI::size(MPI_COMM_WORLD);

  mesh::GhostMode ghost_mode = mesh::GhostMode::shared_facet;

  // TODO: see https://github.com/FEniCS/dolfinx/issues/3358
  //   auto part
  //       = mesh::create_cell_partitioner(ghost_mode,
  //       graph::scotch::partitioner());
  mesh::CellPartitionFunction part
      = [&](MPI_Comm /* comm */, int /* nparts */,
            const std::vector<mesh::CellType>& /* cell_types */,
            const std::vector<std::span<const std::int64_t>>& /* cells */)
  {
    std::vector<std::vector<std::int32_t>> data;
    if (comm_size == 1)
      data = {{0}, {0}, {0}, {0}};
    else if (comm_size == 2)
      data = {{0}, {0}, {0}, {0, 1}, {1, 0}, {1}, {1}, {1}, {1}};
    else if (comm_size == 3)
    {
      data = {{1}, {1}, {1},    {1},    {1, 2}, {2, 1}, {2},
              {2}, {2}, {2, 0}, {0, 2}, {0},    {0},    {0}};
    }
    else
      SKIP("Test only supports <= 3 processes");

    return graph::AdjacencyList<std::int32_t>(data);
  };

  mesh::Mesh<T> mesh = mesh::create_interval<T>(
      MPI_COMM_WORLD, 5 * comm_size - 1, {0., 1.}, ghost_mode, part);

  {
    int comp_result;
    MPI_Comm_compare(mesh.comm(), MPI_COMM_WORLD, &comp_result);
    CHECK(comp_result == MPI_CONGRUENT);
  }

  CHECK(mesh.geometry().dim() == 1);

  std::array<int32_t, 3> expected_local_vertex_count;
  std::array<int32_t, 3> expected_num_ghosts;
  std::vector<T> expected_x;
  std::array<std::vector<std::vector<std::int32_t>>, 3> expected_v_to_e;

  if (comm_size == 1)
  {
    // vertex layout
    //   0 --- 1 --- 2 --- 3 --- 4
    expected_local_vertex_count = {5};
    expected_num_ghosts = {0};

    expected_x = {
        /* v_0 */ 0.0,
        /* v_1 */ 0.25,
        /* v_2 */ 0.5,
        /* v_3 */ 0.75,
        /* v_4 */ 1.0,
    };

    // cell layout
    // x -0- x -1- x -2- x -3- x
    expected_v_to_e[0] = {{0}, {0, 1}, {1, 2}, {2, 3}, {3}};
  }
  else if (comm_size == 2)
  {
    /* clang-format off
      vertex layout
      0 --- 1 --- 2 --- 3 --- 4 --- 5                         (process 0)
      l     l     l     l     l     g
                        6 --- 0 --- 1 --- 2 --- 3 --- 4 --- 5 (process 1)
                        g     g     l     l     l     l     l
      clang-format on */

    expected_local_vertex_count = {4, 6};
    expected_num_ghosts = {2, 1};

    expected_x = {
        /* v_0 */ 0.0,
        /* v_1 */ 1. / 9,
        /* v_2 */ 2. / 9,
        /* v_3 */ 3. / 9,
        /* v_4 */ 4. / 9,
        /* v_5 */ 5. / 9,
        /* v_6 */ 6. / 9,
        /* v_7 */ 7. / 9,
        /* v_8 */ 8. / 9,
        /* v_9 */ 9. / 9,
    };

    /* clang-format off
      cell layout
          x -0- x -1- x -2- x -3- x -4- x                               (process 0)

                            x -5- x -0- x -1- x -2- x -3- x -4- x -5- x (process 1)
      clang-format on */
    expected_v_to_e[0] = {{0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4}};
    expected_v_to_e[1] = {{0, 5}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4}, {5}};
  }
  else if (comm_size == 3)
  {
    /* clang-format off
      vertex layout
      0 --- 1 --- 2 --- 3 --- 4 --- 5 --- 6                                                 (process 1)
      l     l     l     l     l     l     g

                              6 --- 4 --- 0 --- 1 --- 2 --- 3 --- 5 --- 7                   (process 2)
                              g     g     l     l     l     l     l     g

                                                            5 --- 0 --- 1 --- 2 --- 3 --- 4 (process 0)
                                                            g     l     l     l     l     l
      clang-format on */

    expected_local_vertex_count = {5, 6, 4};
    expected_num_ghosts = {1, 1, 4};

    expected_x = {
        /* v_0 */ 0. / 14,
        /* v_1 */ 1. / 14,
        /* v_2 */ 2. / 14,
        /* v_3 */ 3. / 14,
        /* v_4 */ 4. / 14,
        /* v_5 */ 5. / 14,
        /* v_6 */ 6. / 14,
        /* v_7 */ 7. / 14,
        /* v_8 */ 8. / 14,
        /* v_9 */ 9. / 14,
        /* v_10 */ 10. / 14,
        /* v_11 */ 11. / 14,
        /* v_12 */ 12. / 14,
        /* v_13 */ 13. / 14,
        /* v_14 */ 14. / 14,
    };

    /* clang-format off
      vertex layout
      x -0- x -1- x -2- x -3- x -4- x -5- x                                                 (process 1)

                              x -5- x -0- x -1- x -2- x -3- x -4- x -6- x                   (process 2)

                                                            x -4- x -0- x -1- x -2- x -3- x (process 0)
      clang-format on */

    expected_v_to_e[1] = {{0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5}};
    expected_v_to_e[2]
        = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 6}, {0, 5}, {5}, {6}};
    expected_v_to_e[0] = {{0, 4}, {0, 1}, {1, 2}, {2, 3}, {3}, {4}};
  }
  else
  {
    // Test only supports np <= 3
    CHECK(false);
  }

  auto [x, cells]
      = mesh::impl::create_interval_cells<T>({0., 1.}, 5 * comm_size - 1);

  for (std::int64_t i = 0;
       i < mesh.topology()->index_map(0)->size_local()
               + mesh.topology()->index_map(0)->num_ghosts();
       i++)
  {
    CHECK(std::abs(expected_x[mesh.geometry().input_global_indices()[i]]
                   - mesh.geometry().x()[3 * i])
          <= EPS<T>);
    CHECK(std::abs(mesh.geometry().x()[3 * i + 1] - 0.0) <= EPS<T>);
    CHECK(std::abs(mesh.geometry().x()[3 * i + 2] - 0.0) <= EPS<T>);
  }
}

TEMPLATE_TEST_CASE("Rectangle quadrilateral mesh",
                   "[mesh][rectangle][quadrilateral]", float, double)
{
  using T = TestType;

  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::quadrilateral);

  // vertex layout:
  // 1---3
  // |   |
  // |   |
  // |   |
  // 0---2
  std::vector<T> expected_x = {
      /* v_0 */ 0, 0, 0,
      /* v_1 */ 0, 1, 0,
      /* v_2 */ 1, 0, 0,
      /* v_3 */ 1, 1, 0,
  };

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  // edge layout:
  // x-2-x
  // |   |
  // 0   3
  // |   |
  // x-1-x
  mesh.topology()->create_connectivity(1, 0);
  auto e_to_v = mesh.topology()->connectivity(1, 0);
  REQUIRE(e_to_v);
  CHECK_adjacency_list_equal(*e_to_v, {/* e_0 */ {0, 1},
                                       /* e_1 */ {0, 2},
                                       /* e_2 */ {1, 3},
                                       /* e_3 */ {2, 3}});
}

TEMPLATE_TEST_CASE("Rectangle triangle mesh (right)",
                   "[mesh][rectangle][triangle][right]", float, double)
{
  using T = TestType;

  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle,
      mesh::DiagonalType::right);

  // vertex layout:
  // 3---2
  // |  /|
  // | / |
  // |/  |
  // 0---1
  std::vector<T> expected_x = {/* v_0 */ 0, 0, 0,
                               /* v_1 */ 1, 0, 0,
                               /* v_2 */ 1, 1, 0,
                               /* v_3 */ 0, 1, 0};

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  // edge layout:
  // x-4-x
  // |  /|
  // 2 1 3
  // |/  |
  // x-0-x
  mesh.topology()->create_connectivity(1, 0);
  auto e_to_v = mesh.topology()->connectivity(1, 0);
  REQUIRE(e_to_v);

  CHECK_adjacency_list_equal(*e_to_v, {/* e_0 */ {0, 1},
                                       /* e_1 */ {0, 2},
                                       /* e_2 */ {0, 3},
                                       /* e_3 */ {1, 2},
                                       /* e_4 */ {2, 3}});
}

TEMPLATE_TEST_CASE("Rectangle triangle mesh (left)",
                   "[mesh][rectangle][triangle][left]", float, double)
{
  using T = TestType;

  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle,
      mesh::DiagonalType::left);

  // vertex layout:
  // 2---3
  // |\  |
  // | \ |
  // |  \|
  // 0---1
  std::vector<T> expected_x = {
      /* v_0 */ 0, 0, 0,
      /* v_1 */ 1, 0, 0,
      /* v_2 */ 0, 1, 0,
      /* v_3 */ 1, 1, 0,
  };

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  // edge layout:
  // x-4-x
  // |\  |
  // 1 2 3
  // |  \|
  // x-0-x
  mesh.topology()->create_connectivity(1, 0);
  auto e_to_v = mesh.topology()->connectivity(1, 0);
  REQUIRE(e_to_v);
  CHECK_adjacency_list_equal(*e_to_v, {/* e_0 */ {0, 1},
                                       /* e_1 */ {0, 2},
                                       /* e_2 */ {1, 2},
                                       /* e_3 */ {1, 3},
                                       /* e_4 */ {2, 3}});
}

TEMPLATE_TEST_CASE("Rectangle triangle mesh (crossed)",
                   "[mesh][rectangle][triangle][crossed]", float, double)
{
  using T = TestType;

  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle,
      mesh::DiagonalType::crossed);

  // vertex layout:
  // 3---4
  // |\ /|
  // | 2 |
  // |/ \|
  // 0---1
  std::vector<T> expected_x = {
      /* v_0 */ 0,  0,  0,
      /* v_1 */ 1,  0,  0,
      /* v_2 */ .5, .5, 0,
      /* v_3 */ 0,  1,  0,
      /* v_4 */ 1,  1,  0,
  };

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  // edge layout:
  // x-7-x
  // |5 6|
  // 2 x 4
  // |1 3|
  // x-0-x
  mesh.topology()->create_connectivity(1, 0);
  auto e_to_v = mesh.topology()->connectivity(1, 0);
  REQUIRE(e_to_v);

  CHECK_adjacency_list_equal(*e_to_v, {/* e_0 */ {0, 1},
                                       /* e_1 */ {0, 2},
                                       /* e_2 */ {0, 3},
                                       /* e_3 */ {1, 2},
                                       /* e_4 */ {1, 4},
                                       /* e_5 */ {2, 3},
                                       /* e_6 */ {2, 4},
                                       /* e_7 */ {3, 4}});
}

TEMPLATE_TEST_CASE("Box hexahedron mesh", "[mesh][box][hexahedron]", float,
                   double)
{
  using T = TestType;

  mesh::Mesh<T> mesh
      = dolfinx::mesh::create_box<T>(MPI_COMM_SELF, {{{0, 0, 0}, {1, 1, 1}}},
                                     {1, 1, 1}, mesh::CellType::hexahedron);

  // front (z=0) vertex layout
  // 2---3
  // |   |
  // |   |
  // |   |
  // 0---1

  // back (z=1) vertex layout
  // 6---7
  // |   |
  // |   |
  // |   |
  // 4---5
  std::vector<T> expected_x = {
      /* v_0 */ 0, 0, 0,
      /* v_1 */ 1, 0, 0,
      /* v_2 */ 0, 1, 0,
      /* v_3 */ 1, 1, 0,
      /* v_4 */ 0, 0, 1,
      /* v_5 */ 1, 0, 1,
      /* v_6 */ 0, 1, 1,
      /* v_7 */ 1, 1, 1,
  };

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  mesh.topology()->create_connectivity(1, 0);
  auto e_to_v = mesh.topology()->connectivity(1, 0);
  REQUIRE(e_to_v);

  CHECK_adjacency_list_equal(*e_to_v, {/* e_0 */ {0, 1},
                                       /* e_1 */ {0, 2},
                                       /* e_2 */ {0, 4},
                                       /* e_3 */ {1, 3},
                                       /* e_4 */ {1, 5},
                                       /* e_5 */ {2, 3},
                                       /* e_6 */ {2, 6},
                                       /* e_7 */ {3, 7},
                                       /* e_8 */ {4, 5},
                                       /* e_9 */ {4, 6},
                                       /* e_10 */ {5, 7},
                                       /* e_11 */ {6, 7}});

  mesh.topology()->create_connectivity(2, 0);
  auto f_to_v = mesh.topology()->connectivity(2, 0);
  REQUIRE(f_to_v);

  CHECK_adjacency_list_equal(*f_to_v, {
                                          /* f_0 */ {0, 1, 2, 3},
                                          /* f_1 */ {0, 1, 4, 5},
                                          /* f_2 */ {0, 2, 4, 6},
                                          /* f_3 */ {1, 3, 5, 7},
                                          /* f_4 */ {2, 3, 6, 7},
                                          /* f_5 */ {4, 5, 6, 7},
                                      });

  mesh.topology()->create_connectivity(3, 0);
  auto c_to_v = mesh.topology()->connectivity(3, 0);
  REQUIRE(c_to_v);

  CHECK_adjacency_list_equal(*c_to_v, {/* c_0 */ {0, 1, 2, 3, 4, 5, 6, 7}});
}

TEMPLATE_TEST_CASE("Box tetrahedron mesh", "[mesh][box][tetrahedron]", float,
                   double)
{
  using T = TestType;

  mesh::Mesh<T> mesh
      = dolfinx::mesh::create_box<T>(MPI_COMM_SELF, {{{0, 0, 0}, {1, 1, 1}}},
                                     {1, 1, 1}, mesh::CellType::tetrahedron);

  // front (z=0) vertex layout
  // 5---2
  // |  /|
  // | / |
  // |/  |
  // 0---1

  // back (z=1) vertex layout
  // 7---3
  // |  /|
  // | / |
  // |/  |
  // 6---4

  std::vector<T> expected_x = {
      /* v_0 */ 0, 0, 0,
      /* v_1 */ 1, 0, 0,
      /* v_2 */ 1, 1, 0,
      /* v_3 */ 1, 1, 1,
      /* v_4 */ 1, 0, 1,
      /* v_5 */ 0, 1, 0,
      /* v_6 */ 0, 0, 1,
      /* v_7 */ 0, 1, 1,
  };

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  mesh.topology()->create_connectivity(1, 0);
  auto e_to_v = mesh.topology()->connectivity(1, 0);
  REQUIRE(e_to_v);

  CHECK_adjacency_list_equal(*e_to_v, {/* e_0 */ {0, 1},
                                       /* e_1 */ {0, 2},
                                       /* e_2 */ {0, 3},
                                       /* e_3 */ {0, 4},
                                       /* e_4 */ {0, 5},
                                       /* e_5 */ {0, 6},
                                       /* e_6 */ {0, 7},
                                       /* e_7 */ {1, 2},
                                       /* e_8 */ {1, 3},
                                       /* e_9 */ {1, 4},
                                       /* e_10 */ {2, 3},
                                       /* e_11 */ {2, 5},
                                       /* e_12 */ {3, 4},
                                       /* e_13 */ {3, 5},
                                       /* e_14 */ {3, 6},
                                       /* e_15 */ {3, 7},
                                       /* e_16 */ {4, 6},
                                       /* e_17 */ {5, 7},
                                       /* e_18 */ {6, 7}});

  mesh.topology()->create_connectivity(2, 0);
  auto f_to_v = mesh.topology()->connectivity(2, 0);
  REQUIRE(f_to_v);

  CHECK_adjacency_list_equal(*f_to_v, {/* f_0 */ {0, 1, 2},
                                       /* f_1 */ {0, 1, 3},
                                       /* f_2 */ {0, 1, 4},
                                       /* f_3 */ {0, 2, 3},
                                       /* f_4 */ {0, 2, 5},
                                       /* f_5 */ {0, 3, 4},
                                       /* f_6 */ {0, 3, 5},
                                       /* f_7 */ {0, 3, 6},
                                       /* f_8 */ {0, 3, 7},
                                       /* f_9 */ {0, 4, 6},
                                       /* f_10 */ {0, 5, 7},
                                       /* f_11 */ {0, 6, 7},
                                       /* f_12 */ {1, 2, 3},
                                       /* f_13 */ {1, 3, 4},
                                       /* f_14 */ {2, 3, 5},
                                       /* f_15 */ {3, 4, 6},
                                       /* f_16 */ {3, 5, 7},
                                       /* f_17 */ {3, 6, 7}});

  mesh.topology()->create_connectivity(3, 0);
  auto c_to_v = mesh.topology()->connectivity(3, 0);
  REQUIRE(c_to_v);

  CHECK_adjacency_list_equal(*c_to_v, {/* c_0 */ {0, 1, 2, 3},
                                       /* c_1 */ {0, 1, 3, 4},
                                       /* c_2 */ {0, 2, 5, 3},
                                       /* c_3 */ {0, 4, 3, 6},
                                       /* c_4 */ {0, 5, 7, 3},
                                       /* c_5 */ {0, 7, 6, 3}});
}
