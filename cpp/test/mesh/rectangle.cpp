// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "util.h"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/interval.h>
#include <dolfinx/refinement/refine.h>
#include <mpi.h>

using namespace dolfinx;
using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("Rectangle quadrilateral mesh",
                   "[mesh][rectangle][quadrilateral]", float, double)
{
  using T = TestType;

  const std::array<double, 2> lower = {0, 0};
  const std::array<double, 2> upper = {1, 1};
  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {lower, upper}, {1, 1}, mesh::CellType::quadrilateral);

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

  const std::array<double, 2> lower = {0, 0};
  const std::array<double, 2> upper = {1, 1};
  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {lower, upper}, {1, 1}, mesh::CellType::triangle,
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

  const std::array<double, 2> lower = {0, 0};
  const std::array<double, 2> upper = {1, 1};
  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {lower, upper}, {1, 1}, mesh::CellType::triangle,
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

  const std::array<double, 2> lower = {0, 0};
  const std::array<double, 2> upper = {1, 1};
  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {lower, upper}, {1, 1}, mesh::CellType::triangle,
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
