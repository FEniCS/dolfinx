// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <cassert>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <type_traits>

namespace
{
template <typename T>
void test()
{
  using V_runtime = T;
  static_assert(!dolfinx::is_compile_time_v<T, V_runtime>);
  static_assert(dolfinx::is_runtime_v<T, V_runtime>);
  assert((dolfinx::value<T, V_runtime>(V_runtime(1)) == T(1)));

  using V_compile_time = std::integral_constant<T, T(1)>;
  static_assert(dolfinx::is_compile_time_v<T, V_compile_time>);
  static_assert(!dolfinx::is_runtime_v<T, V_compile_time>);
  assert((dolfinx::value<T, V_runtime>(V_compile_time()) == T(1)));
}
} // namespace

TEST_CASE("Test constexpr type", "[constexpr_type]")
{
  test<std::int16_t>();
  test<std::int32_t>();
  test<std::int64_t>();
  test<float>();
  test<double>();
}