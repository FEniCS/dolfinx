// Copyright (C) 2023-2025 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/mdspan.hpp>
#include <complex>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/constexpr_type.h>
#include <type_traits>

namespace dolfinx
{
/// @private This concept is used to constrain the a template type to floating
/// point real or complex types. Note that this concept is different to
/// std::floating_point which does not include std::complex.
template <class T>
concept scalar = std::floating_point<T>
                 || std::is_same_v<T, std::complex<typename T::value_type>>;

/// @private These structs are used to get the float/value type from a
/// template argument, including support for complex types.
template <scalar T, typename = void>
struct scalar_value
{
  /// @internal
  typedef T type;
};
/// @private
template <scalar T>
struct scalar_value<T, std::void_t<typename T::value_type>>
{
  typedef typename T::value_type type;
};
/// @private Convenience typedef
template <scalar T>
using scalar_value_t = typename scalar_value<T>::type;

/// @private mdspan/mdarray namespace
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

/// @private Concept capturing both compile time defined block sizes and runtime
/// ones.
template <typename V>
concept BlockSize = common::ConstexprType<std::int32_t, V>;

/// @private Short notation for a compile time block size.
template <int N>
using BS = std::integral_constant<std::int32_t, N>;

/// @private Retrieves the integral block size of a compile time block size.
template <BlockSize V>
  requires common::is_compile_time_v<std::int32_t, V>
consteval int block_size(V bs)
{
  return common::value<std::int32_t, V>(bs);
}

/// @private Retrieves the integral block size of a runtime block size.
template <BlockSize V>
  requires common::is_runtime_v<std::int32_t, V>
inline int block_size(V bs)
{
  return common::value<std::int32_t, V>(bs);
}

} // namespace dolfinx
