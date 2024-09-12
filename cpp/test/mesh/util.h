// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dolfinx/graph/AdjacencyList.h>

template <typename T>
constexpr auto EPS = std::numeric_limits<T>::epsilon();

template <typename T>
void CHECK_adjacency_list_equal(
    const dolfinx::graph::AdjacencyList<T>& adj_list,
    const std::vector<std::vector<T>>& expected_list)
{
  REQUIRE(static_cast<std::size_t>(adj_list.num_nodes()) == expected_list.size());

  for (T i = 0; i < adj_list.num_nodes(); i++)
    CHECK_THAT(adj_list.links(i), Catch::Matchers::RangeEquals(expected_list[i]));
}
