// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/utils.h>
#include <iterator>
#include <mpi.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dolfinx/graph/partitioners.h>
#include <dolfinx/mesh/generation.h>
#include <vector>

#include "util.h"

using namespace dolfinx;
using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("Interval mesh", "[mesh][interval]", float, double)
{
  using T = TestType;

  mesh::Mesh<T> mesh = mesh::create_interval<T>(MPI_COMM_SELF, 4, {0., 1.});

  {
    int comp_result;
    MPI_Comm_compare(mesh.comm(), MPI_COMM_SELF, &comp_result);
    CHECK(comp_result == MPI_CONGRUENT);
  }

  CHECK(mesh.geometry().dim() == 1);

  // vertex layout
  // 0 --- 1 --- 2 --- 3 --- 4
  std::vector<T> expected_x = {
      /* v_0 */ 0.0,  0.0, 0.0,
      /* v_1 */ 0.25, 0.0, 0.0,
      /* v_2 */ 0.5,  0.0, 0.0,
      /* v_3 */ 0.75, 0.0, 0.0,
      /* v_4 */ 1.0,  0.0, 0.0,
  };

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

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
  int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

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
      data = {{1}, {1}, {1},    {1},    {1, 2}, {2, 1}, {2},
              {2}, {2}, {2, 0}, {0, 2}, {0},    {0},    {0}};
    else
      FAIL("Test only supports <= 3 processes");
    return graph::AdjacencyList<std::int32_t>(std::move(data));
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
  std::array<std::vector<T>, 3> expected_x;
  std::array<std::vector<std::vector<std::int32_t>>, 3> expected_v_to_e;

  if (comm_size == 1)
  {
    // vertex layout
    //   0 --- 1 --- 2 --- 3 --- 4
    expected_local_vertex_count = {5};
    expected_num_ghosts = {0};

    expected_x[0] = {
        /* v_0 */ 0.0,  0.0, 0.0,
        /* v_1 */ 0.25, 0.0, 0.0,
        /* v_2 */ 0.5,  0.0, 0.0,
        /* v_3 */ 0.75, 0.0, 0.0,
        /* v_4 */ 1.0,  0.0, 0.0,
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

    expected_x[0] = {
        /* v_0 */ 0.0,    0.0, 0.0,
        /* v_1 */ 1. / 9, 0.0, 0.0,
        /* v_2 */ 2. / 9, 0.0, 0.0,
        /* v_3 */ 3. / 9, 0.0, 0.0,
        /* v_4 */ 4. / 9, 0.0, 0.0,
        /* v_5 */ 5. / 9, 0.0, 0.0,
    };

    expected_x[1] = {
        /* v_0 */ 4. / 9, 0.0, 0.0,
        /* v_1 */ 5. / 9, 0.0, 0.0,
        /* v_2 */ 6. / 9, 0.0, 0.0,
        /* v_3 */ 7. / 9, 0.0, 0.0,
        /* v_4 */ 8. / 9, 0.0, 0.0,
        /* v_5 */ 9. / 9, 0.0, 0.0,
        /* v_6 */ 3. / 9, 0.0, 0.0,
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

    expected_x[1] = {
        /* v_0 */ 0. / 14, 0.0, 0.0,
        /* v_1 */ 1. / 14, 0.0, 0.0,
        /* v_2 */ 2. / 14, 0.0, 0.0,
        /* v_3 */ 3. / 14, 0.0, 0.0,
        /* v_4 */ 4. / 14, 0.0, 0.0,
        /* v_5 */ 5. / 14, 0.0, 0.0,
        /* v_6 */ 6. / 14, 0.0, 0.0,
    };

    expected_x[2] = {
        /* v_0 */ 6. / 14,  0.0, 0.0,
        /* v_1 */ 7. / 14,  0.0, 0.0,
        /* v_2 */ 8. / 14,  0.0, 0.0,
        /* v_3 */ 9. / 14,  0.0, 0.0,
        /* v_4 */ 5. / 14,  0.0, 0.0,
        /* v_5 */ 10. / 14, 0.0, 0.0,
        /* v_6 */ 4. / 14,  0.0, 0.0,
        /* v_7 */ 11. / 14, 0.0, 0.0,
    };

    expected_x[0] = {
        /* v_0 */ 10. / 14, 0.0, 0.0,
        /* v_1 */ 11. / 14, 0.0, 0.0,
        /* v_2 */ 12. / 14, 0.0, 0.0,
        /* v_3 */ 13. / 14, 0.0, 0.0,
        /* v_4 */ 14. / 14, 0.0, 0.0,
        /* v_5 */ 9. / 14,  0.0, 0.0,
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

  auto vertices = mesh.topology()->index_map(0);

  CHECK(vertices->size_local() == expected_local_vertex_count[rank]);
  CHECK(vertices->num_ghosts() == expected_num_ghosts[rank]);

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x[rank], [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  mesh.topology()->create_connectivity(0, 1);
  CHECK_adjacency_list_equal(*mesh.topology()->connectivity(0, 1),
                             expected_v_to_e[rank]);
}
