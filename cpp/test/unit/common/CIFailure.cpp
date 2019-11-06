// Copyright (C) 2019 Francesco Ballarin
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed Meshes

#include <catch.hpp>
#include <dolfin.h>

using namespace dolfin;

namespace
{
void test_ci_failure()
{

  auto mpi_comm = dolfin::MPI::Comm(MPI_COMM_WORLD);
  int mpi_rank = dolfin::MPI::rank(mpi_comm.comm());

  CHECK(mpi_rank == 0);
}
} // namespace

TEST_CASE("CI failure", "[ci_failure]")
{
  CHECK_NOTHROW(test_ci_failure());
}