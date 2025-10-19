// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <mpi.h>
#include <vector>

#include "dolfinx/common/memory.h"
#include "dolfinx/mesh/Geometry.h"
#include "dolfinx/mesh/generation.h"

using namespace dolfinx::common;

TEMPLATE_TEST_CASE("memory-array", "[memory]", std::int16_t, std::int32_t,
                   std::int64_t, std::uint16_t, std::uint32_t, std::uint64_t,
                   float, double)
//    std::complex<float>, std::complex<double>
{
  std::array<TestType, 10> v;

  std::size_t bytes = 10 * sizeof(TestType);

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

TEMPLATE_TEST_CASE("memory-vector", "[memory]", std::int16_t, std::int32_t,
                   std::int64_t, std::uint16_t, std::uint32_t, std::uint64_t,
                   float, double)
//    std::complex<float>, std::complex<double>
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

TEMPLATE_TEST_CASE("memory-vector-vector", "[memory]", std::int16_t,
                   std::int32_t, std::int64_t, std::uint16_t, std::uint32_t,
                   std::uint64_t, float, double)
//    std::complex<float>, std::complex<double>
{
  std::vector<std::vector<TestType>> v;
  v.reserve(3);
  v.template emplace_back<std::vector<TestType>>({{0, 1, 2}});
  v.template emplace_back<std::vector<TestType>>({{0, 1, 2, 3}});
  v.template emplace_back<std::vector<TestType>>({{0, 1, 2, 3, 4}});

  std::size_t bytes = sizeof(std::vector<std::vector<TestType>>)
                      + 3 * sizeof(std::vector<TestType>)
                      + (3 + 4 + 5) * sizeof(TestType);

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

TEST_CASE("memory-indexmap", "[memory]")
{
  auto im = IndexMap(MPI_COMM_WORLD, 10);
  CHECK(memory(im, byte) > 0);
}

TEMPLATE_TEST_CASE("memory-geometry", "[memory]", float, double)
{
  auto mesh = dolfinx::mesh::create_rectangle<TestType>(
      MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1},
      dolfinx::mesh::CellType::quadrilateral);

  const auto& geo = mesh.geometry();
  CHECK(memory<dolfinx::mesh::Geometry<TestType>>(geo, byte) > 0);
}