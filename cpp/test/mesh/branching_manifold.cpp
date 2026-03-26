// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "catch2/matchers/catch_matchers.hpp"
#include <array>
#include <basix/finite-element.h>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>
#include <optional>
#include <vector>

using namespace dolfinx;

// branching manifold graph G
//
//          (3)
//           |
//           |[2]
//           |
//  (1)-----(0)-----(2) ----- (4)
//      [0]     [1]      [3]
//
// its dual G'
//
//          [2]
//         /   ＼
//        /     ＼
//      [0] --- [1] --- [3]
//

TEST_CASE("dual_graph_branching")
{
  std::vector<mesh::CellType> celltypes{mesh::CellType::interval};
  std::vector<std::int64_t> cells{{0, 1, 0, 2, 0, 3, 2, 4}};

  {
    // default
    auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data]
        = mesh::build_local_dual_graph(celltypes, {cells}, 2);

    CHECK(dual_graph.num_nodes() == 4);

    CHECK(dual_graph.num_links(0) == 2);
    CHECK_THAT(dual_graph.links(0),
               Catch::Matchers::RangeEquals(std::array{1, 2}));

    CHECK(dual_graph.num_links(1) == 3);
    CHECK_THAT(dual_graph.links(3),
               Catch::Matchers::RangeEquals(std::array{1}));

    CHECK_THAT(dual_graph.links(1),
               Catch::Matchers::RangeEquals(std::array{0, 2, 3}));

    CHECK(dual_graph.num_links(2) == 2);
    CHECK_THAT(dual_graph.links(2),
               Catch::Matchers::RangeEquals(std::array{0, 1}));

    CHECK(dual_graph.num_links(3) == 1);
    CHECK_THAT(dual_graph.links(3),
               Catch::Matchers::RangeEquals(std::array{1}));

    CHECK_THAT(unmatched_facets,
               Catch::Matchers::RangeEquals(std::array{1, 3, 4}));

    CHECK(max_vertices_per_facet == 1);

    CHECK_THAT(cell_data, Catch::Matchers::RangeEquals(std::array{0, 2, 3}));
  }

  {
    // max_facet_to_cell_links = 3
    // Note: additionally facet (2) is now considered unmatched
    auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data]
        = mesh::build_local_dual_graph(celltypes, {cells}, 3);

    CHECK(dual_graph.num_nodes() == 4);

    CHECK(dual_graph.num_links(0) == 2);
    CHECK_THAT(dual_graph.links(0),
               Catch::Matchers::RangeEquals(std::array{1, 2}));

    CHECK(dual_graph.num_links(1) == 3);
    CHECK_THAT(dual_graph.links(3),
               Catch::Matchers::RangeEquals(std::array{1}));

    CHECK_THAT(dual_graph.links(1),
               Catch::Matchers::RangeEquals(std::array{0, 2, 3}));

    CHECK(dual_graph.num_links(2) == 2);
    CHECK_THAT(dual_graph.links(2),
               Catch::Matchers::RangeEquals(std::array{0, 1}));

    CHECK(dual_graph.num_links(3) == 1);
    CHECK_THAT(dual_graph.links(3),
               Catch::Matchers::RangeEquals(std::array{1}));

    CHECK_THAT(unmatched_facets,
               Catch::Matchers::RangeEquals(std::array{1, 2, 2, 3, 4}));

    CHECK(max_vertices_per_facet == 1);

    CHECK_THAT(cell_data,
               Catch::Matchers::RangeEquals(std::array{0, 1, 3, 2, 3}));
  }

  {
    // max_facet_to_cell_links = 4 / 5 / std::nullopt
    // Note: all facets are now considered unmatched

    for (auto max_facet_to_cell_links :
         std::array<std::optional<int>, 3>{4, 5, std::nullopt})
    {

      auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data]
          = mesh::build_local_dual_graph(celltypes, {cells},
                                         max_facet_to_cell_links);

      CHECK(dual_graph.num_nodes() == 4);

      CHECK(dual_graph.num_links(0) == 2);
      CHECK_THAT(dual_graph.links(0),
                 Catch::Matchers::RangeEquals(std::array{1, 2}));

      CHECK(dual_graph.num_links(1) == 3);
      CHECK_THAT(dual_graph.links(3),
                 Catch::Matchers::RangeEquals(std::array{1}));

      CHECK_THAT(dual_graph.links(1),
                 Catch::Matchers::RangeEquals(std::array{0, 2, 3}));

      CHECK(dual_graph.num_links(2) == 2);
      CHECK_THAT(dual_graph.links(2),
                 Catch::Matchers::RangeEquals(std::array{0, 1}));

      CHECK(dual_graph.num_links(3) == 1);
      CHECK_THAT(dual_graph.links(3),
                 Catch::Matchers::RangeEquals(std::array{1}));

      CHECK_THAT(unmatched_facets, Catch::Matchers::RangeEquals(
                                       std::array{0, 0, 0, 1, 2, 2, 3, 4}));

      CHECK(max_vertices_per_facet == 1);

      CHECK_THAT(cell_data, Catch::Matchers::RangeEquals(
                                std::array{0, 1, 2, 0, 1, 3, 2, 3}));
    }
  }
}

// branching manifold graph G
//
//          (2)
//     [2] /   ＼ [1]
//        /     ＼
//      (0) --- (1)
//          [0]
//
// its dual G'
//
//     [2] --- [1]
//       ＼    /
//        ＼  /
//         [0]
//
TEST_CASE("dual_graph_self_dual")
{
  std::vector<mesh::CellType> celltypes{mesh::CellType::interval};
  std::vector<std::int64_t> cells{{0, 1, 1, 2, 2, 0}};

  for (auto max_facet_to_cell_links :
       std::array<std::optional<int>, 3>{3, 4, std::nullopt})
  {
    auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data]
        = mesh::build_local_dual_graph(celltypes, {cells},
                                       max_facet_to_cell_links);

    CHECK(max_vertices_per_facet == 1);
    CHECK(dual_graph.num_nodes() == 3);

    CHECK(dual_graph.num_links(0) == 2);

    CHECK_THAT(dual_graph.links(0),
               Catch::Matchers::RangeEquals(std::array{2, 1}));

    CHECK(dual_graph.num_links(1) == 2);
    CHECK_THAT(dual_graph.links(1),
               Catch::Matchers::RangeEquals(std::array{0, 2}));

    CHECK(dual_graph.num_links(2) == 2);
    CHECK_THAT(dual_graph.links(2),
               Catch::Matchers::RangeEquals(std::array{0, 1}));

    CHECK_THAT(unmatched_facets,
               Catch::Matchers::RangeEquals(std::array{0, 0, 1, 1, 2, 2}));

    CHECK_THAT(cell_data,
               Catch::Matchers::RangeEquals(std::array{0, 2, 0, 1, 1, 2}));
  }
}

// Parallel branching manifold graph G ('⟷': indicates the process boundary)
//
//          (3)
//           |
//           |[2]
//           |
//  (1)-----(0)  ⟷  (0)-----(2)-----(4)
//      [0]              [1]     [3]
//
//
// its local dual graphs
//
//          [1]
//         /
//        /
//      [0]     ⟷  [0] --- [1]
//
//
// its (global) dual G'
//
//          [1]
//         /   ＼
//        /     ＼
//      [0] --- [2] --- [3]
//
TEST_CASE("dual_graph_branching_parallel")
{
  auto comm = MPI_COMM_WORLD;

  if (dolfinx::MPI::size(comm) != 2)
    SKIP("Only supports two processes.");

  std::vector<mesh::CellType> celltypes{mesh::CellType::interval};

  std::vector<std::int64_t> cells;
  if (dolfinx::MPI::rank(comm) == 0)
    cells = {{0, 1, 0, 3}};
  else
    cells = {{0, 2, 2, 4}};

  {
    // Check local dual graphs.

    auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data]
        = mesh::build_local_dual_graph(celltypes, {cells}, 3);

    CHECK(max_vertices_per_facet == 1);

    CHECK(dual_graph.num_nodes() == 2);

    CHECK(dual_graph.num_links(0) == 1);
    CHECK_THAT(dual_graph.links(0),
               Catch::Matchers::RangeEquals(std::array{1}));
    CHECK(dual_graph.num_links(1) == 1);
    CHECK_THAT(dual_graph.links(1),
               Catch::Matchers::RangeEquals(std::array{0}));
    if (dolfinx::MPI::rank(comm) == 0)
    {
      CHECK_THAT(unmatched_facets,
                 Catch::Matchers::RangeEquals(std::array{0, 0, 1, 3}));
      CHECK_THAT(cell_data,
                 Catch::Matchers::RangeEquals(std::array{0, 1, 0, 1}));
    }
    else
    {
      CHECK_THAT(unmatched_facets,
                 Catch::Matchers::RangeEquals(std::array{0, 2, 2, 4}));
      CHECK_THAT(cell_data,
                 Catch::Matchers::RangeEquals(std::array{0, 0, 1, 1}));
    }
  }

  auto dual_graph = mesh::build_dual_graph(
      comm, celltypes, std::vector<std::span<const std::int64_t>>{cells}, 3);

  if (dolfinx::MPI::rank(comm) == 0)
  {
    CHECK(dual_graph.num_nodes() == 2);

    CHECK(dual_graph.num_links(0) == 2);
    CHECK_THAT(dual_graph.links(0),
               Catch::Matchers::RangeEquals(std::array{1, 2}));
    CHECK(dual_graph.num_links(1) == 2);
    CHECK_THAT(dual_graph.links(1),
               Catch::Matchers::RangeEquals(std::array{0, 2}));
  }
  else
  {
    CHECK(dual_graph.num_nodes() == 2);

    CHECK(dual_graph.num_links(0) == 3);
    CHECK_THAT(dual_graph.links(0),
               Catch::Matchers::RangeEquals(std::array{0, 1, 3}));
    CHECK(dual_graph.num_links(1) == 1);
    CHECK_THAT(dual_graph.links(1),
               Catch::Matchers::RangeEquals(std::array{2}));
  }
}
