// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/defines.h>

TEST_CASE("CUDA support", "[cuda]")
{
#ifdef HAS_CUDA
  STATIC_REQUIRE(dolfinx::has_cuda());
#else
  STATIC_REQUIRE_FALSE(dolfinx::has_cuda());
#endif
}
