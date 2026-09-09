// Copyright (C) 2026 Paul T. Kühner and Jack S. Hale
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
#include <mpi.h>
#include <span>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::refinement;

TEMPLATE_TEST_CASE("Mark maximum empty", "[refinement][mark][maximum]", double,
                   float)
{
  common::IndexMap im(MPI_COMM_WORLD, 0);
  std::vector<TestType> v;
  auto indices = mark_maximum(std::span<const TestType>(v), im, .5);
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
  common::IndexMap im(comm, local_size, ghosts, owners);

  std::vector<TestType> v(im.size_local() + im.num_ghosts());
  if (rank == 0)
  {
    CHECK(v.size() == static_cast<std::size_t>(size));
    for (int i = 0; i < size; i++)
      v[i] = i;
  }
  else
  {
    CHECK(v.size() == 1);
    // Poison the ghost slot: if the local max were wrongly computed over
    // ghosts too, this would inflate the (globally reduced) threshold and
    // the checks below would fail.
    v[0] = static_cast<TestType>(1000);
  }

  TestType theta = 0.5;
  auto indices = mark_maximum(std::span<const TestType>(v), im, theta);

  CHECK(std::ranges::all_of(
      indices, [&v](auto e)
      { return (0 <= e) && (e <= static_cast<std::int32_t>(v.size())); }));

  TestType max = size - 1;
  auto mark = [&theta, &max](auto e) { return e > theta * max; };

  CHECK(std::ranges::count_if(v, mark)
        == static_cast<std::int32_t>(indices.size()));

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(v.size()); ++i)
  {
    bool expect_marked = mark(v[i]);
    bool marked = std::ranges::find(indices, i) != indices.end();
    CHECK(expect_marked == marked);
  }
}
