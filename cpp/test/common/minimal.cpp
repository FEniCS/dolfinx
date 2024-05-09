// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <dolfinx/common/MPI.h>

using namespace dolfinx;

namespace
{
void minimal()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
}

TEST_CASE("Minimal", "[minimal]")
{
  CHECK_NOTHROW(minimal());
}
}
