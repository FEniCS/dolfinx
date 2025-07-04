// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "catch2/matchers/catch_matchers.hpp"
#include <algorithm>
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
#include <iterator>
#include <mpi.h>
#include <optional>
#include <ostream>
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
        = mesh::build_local_dual_graph(celltypes, {cells});

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
    // matched_facet_cell_count = 2
    // Note: additioanlly facet (2) is now considered unmatched
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
               Catch::Matchers::RangeEquals(std::array{1, 2, 3, 4}));

    CHECK(max_vertices_per_facet == 1);

    CHECK_THAT(cell_data, Catch::Matchers::RangeEquals(std::array{0, 1, 2, 3}));
  }

  {
    // matched_facet_cell_count = 3 / std::nullopt
    // Note: all facets are now considered unmatched

    for (auto matched_facet_cell_count :
         std::array<std::optional<int>, 2>{3, std::nullopt})
    {

      auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data]
          = mesh::build_local_dual_graph(celltypes, {cells},
                                         matched_facet_cell_count);

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
                 Catch::Matchers::RangeEquals(std::array{0, 1, 2, 3, 4}));

      CHECK(max_vertices_per_facet == 1);

      CHECK_THAT(cell_data,
                 Catch::Matchers::RangeEquals(std::array{0, 0, 1, 2, 3}));
    }
  }
}

// branching manifold graph G
//
//          (2)
//     [1] /   ＼ [0]
//        /     ＼
//      (0) --- (1)
//          [2]
//
// its dual G'
//
//          [2]
//         /   ＼
//        /     ＼
//      [0] --- [1]
//
TEST_CASE("dual_graph_self_dual")
{
  std::vector<mesh::CellType> celltypes{mesh::CellType::interval};
  std::vector<std::int64_t> cells{{0, 1, 1, 2, 2, 0}};

  auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data]
      = mesh::build_local_dual_graph(celltypes, {cells}, 2);

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
             Catch::Matchers::RangeEquals(std::array{0, 1, 2}));

  CHECK_THAT(cell_data, Catch::Matchers::RangeEquals(std::array{0, 0, 1}));
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
//          [2]
//         /
//        /
//      [0]     ⟷  [1] --- [3]
//
//
// its (global) dual G'
//
//          [2]
//         /   ＼
//        /     ＼
//      [0] --- [1] --- [3]
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
        = mesh::build_local_dual_graph(celltypes, {cells}, 2);

    CHECK(max_vertices_per_facet == 1);
    CHECK(unmatched_facets.size() == 3);

    CHECK(dual_graph.num_nodes() == 2);

    CHECK(dual_graph.num_links(0) == 1);
    CHECK_THAT(dual_graph.links(0),
               Catch::Matchers::RangeEquals(std::array{1}));
    CHECK(dual_graph.num_links(1) == 1);
    CHECK_THAT(dual_graph.links(1),
               Catch::Matchers::RangeEquals(std::array{0}));
    CHECK_THAT(cell_data, Catch::Matchers::RangeEquals(std::array{0, 0, 1}));
    if (dolfinx::MPI::rank(comm) == 0)
    {
      CHECK_THAT(unmatched_facets,
                 Catch::Matchers::RangeEquals(std::array{0, 1, 3}));
    }
    else
    {
      CHECK_THAT(unmatched_facets,
                 Catch::Matchers::RangeEquals(std::array{0, 2, 4}));
    }
    // all facets unmatched
    // std::cout << dolfinx::MPI::rank(comm) << " unmatched_facets:";
    // for (auto e : unmatched_facets)
    //   std::cout << e << ", ";
    // std::cout << std::endl;

    // std::cout << dolfinx::MPI::rank(comm) << " cell_data:";
    // for (auto e : cell_data)
    //   std::cout << e << ", ";
    // std::cout << std::endl;
    // std::cout << dolfinx::MPI::rank(comm) << " " << dual_graph.str() <<
    // std::endl;
  }
  auto dual_graph = mesh::build_dual_graph(
      comm, celltypes, std::vector<std::span<const std::int64_t>>{cells}, 2);
  std::cout << dolfinx::MPI::rank(comm) << " " << dual_graph.str() << std::endl;

  // int max_links = 0;
  // for (std::int32_t c = 0; c < dual_graph.num_nodes(); c++)
  //   max_links = std::max(max_links, dual_graph.num_links(c));

  // // One process has 2 one 3.
  // CHECK(((max_links == 2) or (max_links == 3)));

  // int sum = 0;
  // MPI_Allreduce(&max_links, &sum, 1, MPI_INT, MPI_SUM, comm);
  // CHECK(sum == 5);
}
