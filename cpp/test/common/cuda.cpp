// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/defines.h>

#ifdef HAS_CUDA
#include "cuda_nvcc.h"
#include <mpi.h>
#endif

TEST_CASE("CUDA support", "[cuda]")
{
#ifdef HAS_CUDA
  STATIC_REQUIRE(dolfinx::has_cuda());

  // DOLFINx code compiled by nvcc (cuda_nvcc.cu)
  CHECK(dolfinx_test::cuda_nvcc_host_code(MPI_COMM_WORLD));
#else
  STATIC_REQUIRE_FALSE(dolfinx::has_cuda());
#endif
}
