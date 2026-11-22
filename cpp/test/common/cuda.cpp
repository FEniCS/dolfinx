// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/defines.h>

#ifdef HAS_CUDA
#include <cuda_runtime_api.h>
#endif

TEST_CASE("CUDA support", "[cuda]")
{
#ifdef HAS_CUDA
  STATIC_REQUIRE(dolfinx::has_cuda());

  // Queries the linked runtime; no GPU or driver needed.
  int version = 0;
  REQUIRE(cudaRuntimeGetVersion(&version) == cudaSuccess);
  CHECK(version > 0);
#else
  STATIC_REQUIRE_FALSE(dolfinx::has_cuda());
#endif
}
