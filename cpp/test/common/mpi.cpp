// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/MPI.h>
#include <mpi.h>
#include <utility>

using namespace dolfinx;

namespace
{
/// Start a non-blocking operation, so that the request is not null
void start(MPI::Request& request)
{
  REQUIRE(MPI_Ibarrier(MPI_COMM_SELF, &request.request()) == MPI_SUCCESS);
  REQUIRE(request.request() != MPI_REQUEST_NULL);
}

/// Complete a started operation
void finish(MPI::Request& request)
{
  REQUIRE(MPI_Wait(&request.request(), MPI_STATUS_IGNORE) == MPI_SUCCESS);
}
} // namespace

TEST_CASE("MPI request is null by default", "[mpi_request]")
{
  MPI::Request request;
  CHECK(request.request() == MPI_REQUEST_NULL);
}

TEST_CASE("MPI request transfers on move", "[mpi_request]")
{
  SECTION("move construction")
  {
    MPI::Request request;
    start(request);
    const MPI_Request handle = request.request();

    MPI::Request moved(std::move(request));

    // The request must move, so that only one holder names it
    CHECK(moved.request() == handle);
    CHECK(request.request() == MPI_REQUEST_NULL);
    finish(moved);
  }

  SECTION("move assignment")
  {
    MPI::Request request;
    start(request);
    const MPI_Request handle = request.request();

    MPI::Request moved;
    moved = std::move(request);

    CHECK(moved.request() == handle);
    CHECK(request.request() == MPI_REQUEST_NULL);
    finish(moved);
  }
}

TEST_CASE("MPI request is not inherited by a copy", "[mpi_request]")
{
  MPI::Request request;
  start(request);

  SECTION("copy construction")
  {
    MPI::Request copy(request);
    CHECK(copy.request() == MPI_REQUEST_NULL);
  }

  SECTION("copy assignment")
  {
    MPI::Request copy;
    copy = request;
    CHECK(copy.request() == MPI_REQUEST_NULL);
  }

  // The original keeps the request, and must still complete it
  CHECK(request.request() != MPI_REQUEST_NULL);
  finish(request);
}
