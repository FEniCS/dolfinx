// Copyright (C) 2026 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/refinement/mark.h>
#include <memory>
#include <mpi.h>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::refinement;

TEMPLATE_TEST_CASE("Mark maximum empty", "[refinement][mark][maximum]", double,
                   float)
{
  dolfinx::la::Vector<TestType> marker(
      std::make_shared<common::IndexMap>(MPI_COMM_WORLD, 0), 1);
  auto indices = mark_maximum<TestType>(marker, .5);
  CHECK(indices.size() == 0);
}

TEMPLATE_TEST_CASE("Mark maximum", "[refinement][mark][maximum]", double, float)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = dolfinx::MPI::rank(comm);
  int size = dolfinx::MPI::size(comm);

  // vec: comm size entries owned by rank 0; each other process (rank>0) gets
  // one as ghost
  std::int32_t local_size = (rank == 0) ? size : 0;
  std::vector<std::int64_t> ghosts = (rank == 0)
                                         ? std::vector<std::int64_t>{}
                                         : std::vector<std::int64_t>{rank};
  std::vector<int> owners
      = (rank == 0) ? std::vector<int>{} : std::vector<int>{0};
  dolfinx::la::Vector<TestType> marker(
      std::make_shared<common::IndexMap>(comm, local_size, ghosts, owners), 1);

  if (rank == 0)
  {
    CHECK(marker.array().size() == static_cast<std::size_t>(size));
    for (int i = 0; i < size; i++)
      marker.array()[i] = i;
  }
  else
    CHECK(marker.array().size() == 1);

  TestType theta = 0.5;
  auto indices = mark_maximum<TestType>(marker, theta);

  CHECK(std::ranges::all_of(indices,
                            [&marker](auto e)
                            {
                              return (0 <= e)
                                     && (e <= static_cast<std::int32_t>(
                                             marker.array().size()));
                            }));

  TestType max = size - 1;
  auto mark = [&theta, &max](auto e) { return e > theta * max; };

  CHECK(std::ranges::count_if(marker.array(), mark)
        == static_cast<std::int32_t>(indices.size()));

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(marker.array().size());
       ++i)
  {
    bool expect_marked = mark(marker.array()[i]);
    bool marked = std::ranges::find(indices, i) != indices.end();
    CHECK(expect_marked == marked);
  }
}
