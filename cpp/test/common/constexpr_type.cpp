// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <cassert>
#include <cstdint>
#include <dolfinx/common/constexpr_type.h>
#include <type_traits>

using namespace dolfinx::common;

namespace
{
template <typename T>
void test()
{
  using V_runtime = T;
  static_assert(!is_compile_time_v<T, V_runtime>);
  static_assert(is_runtime_v<T, V_runtime>);
  assert((value<T, V_runtime>(V_runtime(1)) == T(1)));

  using V_compile_time = std::integral_constant<T, T(1)>;
  static_assert(is_compile_time_v<T, V_compile_time>);
  static_assert(!is_runtime_v<T, V_compile_time>);
  static_assert((value<T, V_compile_time>(V_compile_time()) == T(1)));
}
} // namespace

TEST_CASE("Test constexpr type", "[constexpr_type]")
{
  test<std::int16_t>();
  test<std::int32_t>();
  test<std::int64_t>();

// is C++ 20, but some compilers do not fully support, see
// https://en.cppreference.com/w/cpp/compiler_support/20#cpp_nontype_template_args_201911L
#if defined(__cpp_nontype_template_args)
  test<float>();
  test<double>();
#endif
}