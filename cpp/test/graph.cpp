// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <dolfinx/graph/AdjacencyList.h>
#include <vector>

using namespace dolfinx;

namespace
{
void test_adjacency_list_create()
{
  std::vector<std::int32_t> edges{1, 2, 0, 0, 1};
  std::vector<std::int32_t> offsets{0, 2, 3, 5};
  graph::AdjacencyList g0(edges, offsets);

  graph::AdjacencyList<std::int32_t> g2(10);
  graph::AdjacencyList<std::int32_t, int> g3(10);

  auto g4 = graph::regular_adjacency_list<int>(edges, 1);
  // auto g2 = graph::regular_adjacency_list(edges, 1);

  CHECK(std::ranges::equal(g0.links(0), std::vector<std::int32_t>{1, 2}));
  CHECK(std::ranges::equal(g0.links(1), std::vector<std::int32_t>{0}));
  CHECK(std::ranges::equal(g0.links(2), std::vector<std::int32_t>{0, 1}));

  std::vector<std::int64_t> node_data{-1, 5, -20};
  graph::AdjacencyList g1(edges, offsets, node_data);
  CHECK(std::ranges::equal(g1.node_data().value(), node_data));
}
} // namespace

TEST_CASE("AdjacencyList create")
{
  CHECK_NOTHROW(test_adjacency_list_create());
}

#endif
