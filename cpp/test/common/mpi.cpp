// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/MPI.h>
#include <mpi.h>
#include <utility>

namespace
{
/// Start a non-blocking operation, so that the request is not null
void start(dolfinx::MPI::Request& request)
{
  REQUIRE(MPI_Ibarrier(MPI_COMM_SELF, &request.request()) == MPI_SUCCESS);
  REQUIRE(request.request() != MPI_REQUEST_NULL);
}

/// Complete a started operation
void finish(dolfinx::MPI::Request& request)
{
  REQUIRE(MPI_Wait(&request.request(), MPI_STATUS_IGNORE) == MPI_SUCCESS);
}
} // namespace

TEST_CASE("MPI request is null by default", "[mpi_request]")
{
  dolfinx::MPI::Request request;
  CHECK(request.request() == MPI_REQUEST_NULL);
}

TEST_CASE("MPI request transfers on move", "[mpi_request]")
{
  SECTION("move construction")
  {
    dolfinx::MPI::Request request;
    start(request);
    const MPI_Request handle = request.request();

    dolfinx::MPI::Request moved(std::move(request));

    // The request must move, so that only one holder names it
    CHECK(moved.request() == handle);
    CHECK(request.request() == MPI_REQUEST_NULL);
    finish(moved);
  }

  SECTION("move assignment")
  {
    dolfinx::MPI::Request request;
    start(request);
    const MPI_Request handle = request.request();

    dolfinx::MPI::Request moved;
    moved = std::move(request);

    CHECK(moved.request() == handle);
    CHECK(request.request() == MPI_REQUEST_NULL);
    finish(moved);
  }
}

TEST_CASE("MPI request is not inherited by a copy", "[mpi_request]")
{
  dolfinx::MPI::Request request;
  start(request);

  SECTION("copy construction")
  {
    dolfinx::MPI::Request copy(request);
    CHECK(copy.request() == MPI_REQUEST_NULL);
  }

  SECTION("copy assignment")
  {
    dolfinx::MPI::Request copy;
    copy = request;
    CHECK(copy.request() == MPI_REQUEST_NULL);
  }

  // The original keeps the request, and must still complete it
  CHECK(request.request() != MPI_REQUEST_NULL);
  finish(request);
}
