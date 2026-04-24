// Copyright (C) 2026 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <dolfinx/common/MPI.h>
#include <dolfinx/refinement/mark.h>
#include <mpi.h>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::refinement;

TEMPLATE_TEST_CASE("Mark maximum empty", "[refinement][mark][maximum]", double,
                   float)
{
  std::vector<TestType> marker;
  auto indices = mark_maximum<TestType>(marker, .5, MPI_COMM_WORLD);
  CHECK(indices.size() == 0);
}

TEMPLATE_TEST_CASE("Mark maximum ones", "[refinement][mark][maximum]", double,
                   float)
{
  std::vector<TestType> marker(10, 1.0);
  auto indices = mark_maximum<TestType>(marker, 1.0, MPI_COMM_WORLD);
  CHECK(indices.size() == 10);
}

TEMPLATE_TEST_CASE("Mark maximum", "[refinement][mark][maximum]", double, float)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  std::vector<TestType> marker;
  marker.reserve(10);
  for (std::size_t i = 0; i < 10; i++)
    marker.push_back(10 * dolfinx::MPI::rank(comm) + i);

  TestType theta = 0.5;
  auto indices = mark_maximum<TestType>(marker, theta, comm);

  CHECK(std::ranges::all_of(
      indices, [&](auto e)
      { return (0 <= e) && (e <= static_cast<std::int32_t>(marker.size())); }));

  TestType max = dolfinx::MPI::size(comm) * 10 - 1;
  auto mark = [=](auto e) { return e >= theta * max; };

  CHECK(std::ranges::count_if(marker, mark)
        == static_cast<std::int32_t>(indices.size()));

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(marker.size()); ++i)
  {
    bool expect_marked = mark(marker[i]);
    bool marked = std::ranges::find(indices, i) != indices.end();
    CHECK(expect_marked == marked);
  }
}
