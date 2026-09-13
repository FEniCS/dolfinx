// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/local_range.h>
#include <mpi.h>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

TEST_CASE("local_range bad arguments", "[local_range]")
{
  CHECK_THROWS_AS(common::local_range(-1, 10, 1), std::invalid_argument);
  CHECK_THROWS_AS(common::local_range(0, -1, 1), std::invalid_argument);
  CHECK_THROWS_AS(common::local_range(0, 10, 0), std::invalid_argument);
  CHECK_THROWS_AS(common::local_range(0, 10, -1), std::invalid_argument);
}

TEST_CASE("distribute_data bad arguments", "[distribute_data]")
{
  std::vector<std::int64_t> indices;
  std::vector<double> x;
  CHECK_THROWS_AS(
      MPI::distribute_data(MPI_COMM_WORLD, indices, MPI_COMM_WORLD, x, 0),
      std::invalid_argument);
  CHECK_THROWS_AS(
      MPI::distribute_data(MPI_COMM_WORLD, indices, MPI_COMM_WORLD, x, -1),
      std::invalid_argument);
}
