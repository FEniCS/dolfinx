// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cuda_nvcc.h"
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

TEST_CASE("DOLFINx code compiled by nvcc", "[cuda]")
{
  CHECK(dolfinx_test::cuda_nvcc_host_code(MPI_COMM_WORLD));
}
