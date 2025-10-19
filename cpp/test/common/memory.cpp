// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <vector>

#include "dolfinx/common/memory.h"

using namespace dolfinx::common;

TEMPLATE_TEST_CASE("memory-vector", "[memory]", std::int16_t, std::int32_t,
                   std::int64_t, std::uint16_t, std::uint32_t, std::uint64_t,
                   float, double, std::complex<float>, std::complex<double>)
{
  std::vector<TestType> v;
  v.reserve(10);

  std::size_t bytes = sizeof(std::vector<TestType>) + 10 * sizeof(TestType);

  CHECK(memory(v, byte) == bytes);

  CHECK(memory(v, kilobyte)
        == Catch::Approx(static_cast<double>(bytes) / kilobyte));
  CHECK(memory(v, megabyte)
        == Catch::Approx(static_cast<double>(bytes) / megabyte));
  CHECK(memory(v, gigabyte)
        == Catch::Approx(static_cast<double>(bytes) / gigabyte));
  CHECK(memory(v, terabyte)
        == Catch::Approx(static_cast<double>(bytes) / terabyte));
}
