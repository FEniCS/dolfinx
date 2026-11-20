// Copyright (C) 2025-2026 Paul T. Kühner
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
#include <dolfinx/mesh/graphbuild.h>
#include <dolfinx/mesh/utils.h>
#include <limits>
#include <mpi.h>
#include <optional>
#include <span>
#include <stdexcept>
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
    auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data, _ew,
          _uw] = mesh::build_local_dual_graph(celltypes, {cells}, 2, 1, {});

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
    auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data, _ew,
          _uw] = mesh::build_local_dual_graph(celltypes, {cells}, 3, 1, {});

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

      auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data,
            _ew, _uw]
          = mesh::build_local_dual_graph(celltypes, {cells},
                                         max_facet_to_cell_links, 1, {});

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
    auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data, _ew,
          _uw] = mesh::build_local_dual_graph(celltypes, {cells},
                                              max_facet_to_cell_links, 1, {});

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

    auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data, _ew,
          _uw] = mesh::build_local_dual_graph(celltypes, {cells}, 3, 1, {});

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

TEST_CASE("local_dual_graph_facet_weights")
{
  const std::vector<mesh::CellType> types{mesh::CellType::interval};
  const std::vector<std::int64_t> cells{0, 1, 0, 2, 0, 3, 2, 4};
  const std::vector<std::int32_t> values{2, 11, 5, 8, 12, 13, 10, 17};
  const std::array<std::span<const std::int32_t>, 1> input{values};
  for (int threads : {1, 2, 8})
  {
    auto [graph, facets, width, attached, weights, unmatched]
        = mesh::build_local_dual_graph(types, {cells}, std::nullopt, threads,
                                       input);
    REQUIRE(graph.num_nodes() == 4);
    REQUIRE(weights.size() == graph.array().size());
    REQUIRE(unmatched.size() == attached.size());
    CHECK(width == 1);
    for (std::int32_t c = 0; c < graph.num_nodes(); ++c)
      for (std::int32_t p = graph.offsets()[c]; p < graph.offsets()[c + 1]; ++p)
        CHECK(weights[p] == ((c == 3 || graph.array()[p] == 3) ? 9 : 6));
    // Unmatched entries retain each cell's original contribution, not the mean.
    for (std::size_t i = 0; i < attached.size(); ++i)
    {
      const std::int32_t c = attached[i];
      const int f = cells[2 * c] == facets[i] ? 0 : 1;
      CHECK(unmatched[i] == values[2 * c + f]);
    }
    auto [plain, pf, pw, pc, no_weights, no_unmatched]
        = mesh::build_local_dual_graph(types, {cells}, std::nullopt, threads,
                                       {});
    CHECK_THAT(plain.array(), Catch::Matchers::RangeEquals(graph.array()));
    CHECK(no_weights.capacity() == 0);
    CHECK(no_unmatched.capacity() == 0);
  }
}

TEST_CASE("local_dual_graph_mixed_facet_weights")
{
  const std::vector<mesh::CellType> types{mesh::CellType::triangle,
                                          mesh::CellType::quadrilateral};
  const std::vector<std::int64_t> triangles{0, 1, 2};
  const std::vector<std::int64_t> quads{1, 3, 2, 4};
  const std::vector<std::int32_t> tw{4, 11, 12}, qw{20, 8, 22, 23};
  const std::array<std::span<const std::int32_t>, 2> input{tw, qw};
  for (int threads : {1, 3})
  {
    auto [graph, facets, width, attached, weights, unmatched]
        = mesh::build_local_dual_graph(types, {triangles, quads}, 2, threads,
                                       input);
    REQUIRE(graph.num_nodes() == 2);
    CHECK_THAT(graph.links(0), Catch::Matchers::RangeEquals(std::array{1}));
    CHECK_THAT(graph.links(1), Catch::Matchers::RangeEquals(std::array{0}));
    CHECK_THAT(weights, Catch::Matchers::RangeEquals(std::array{6, 6}));
    CHECK(unmatched.size() == 5);
    auto [plain, pf, pw, pc, _ew, _uw] = mesh::build_local_dual_graph(
        types, {triangles, quads}, 2, threads, {});
    CHECK(plain.num_nodes() == 2);
    CHECK_THAT(plain.array(), Catch::Matchers::RangeEquals(graph.array()));
  }
}

TEST_CASE("local_dual_graph_weight_validation")
{
  const std::vector<mesh::CellType> types{mesh::CellType::interval};
  const std::vector<std::int64_t> cells{0, 1, 0, 2};
  const std::int32_t max = std::numeric_limits<std::int32_t>::max();
  const std::vector<std::int32_t> values(4, max);
  const std::array<std::span<const std::int32_t>, 1> input{values};
  auto [graph, facets, width, attached, weights, unmatched]
      = mesh::build_local_dual_graph(types, {cells}, 2, 1, input);
  CHECK_THAT(weights, Catch::Matchers::RangeEquals(std::array{max, max}));
  CHECK_THROWS_AS(
      mesh::build_local_dual_graph(
          types, {cells}, 2, 1,
          std::array<std::span<const std::int32_t>, 2>{values, values}),
      std::invalid_argument);
  CHECK_THROWS_AS(
      mesh::build_local_dual_graph(types, {cells}, 2, 1,
                                   std::array<std::span<const std::int32_t>, 1>{
                                       std::span(values).first(3)}),
      std::invalid_argument);
  const std::array<std::span<const std::int32_t>, 1> empty_weights{};
  auto [empty, ef, ew, ec, eweights, eunmatched]
      = mesh::build_local_dual_graph(types, {{}}, 2, 1, empty_weights);
  CHECK(empty.num_nodes() == 0);
  CHECK(eweights.capacity() == 0);
  CHECK(eunmatched.capacity() == 0);
}
