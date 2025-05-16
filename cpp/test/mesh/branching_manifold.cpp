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
  auto [dual_graph, unmatched_facets, max_vertices_per_facet, cell_data]
      = mesh::build_local_dual_graph(celltypes, {cells});

  CHECK(dual_graph.num_nodes() == 4);

  CHECK(dual_graph.num_links(0) == 2);
  CHECK_THAT(dual_graph.links(0),
             Catch::Matchers::RangeEquals(std::array{1, 2}));

  CHECK(dual_graph.num_links(1) == 3);
  CHECK_THAT(dual_graph.links(3), Catch::Matchers::RangeEquals(std::array{1}));

  CHECK_THAT(dual_graph.links(1),
             Catch::Matchers::RangeEquals(std::array{0, 2, 3}));

  CHECK(dual_graph.num_links(2) == 2);
  CHECK_THAT(dual_graph.links(2),
             Catch::Matchers::RangeEquals(std::array{0, 1}));

  CHECK(dual_graph.num_links(3) == 1);
  CHECK_THAT(dual_graph.links(3), Catch::Matchers::RangeEquals(std::array{1}));

  CHECK_THAT(unmatched_facets,
             Catch::Matchers::RangeEquals(std::array{1, 3, 4}));

  CHECK(max_vertices_per_facet == 1);

  CHECK_THAT(cell_data, Catch::Matchers::RangeEquals(std::array{0, 2, 3}));
}
