// Copyright (C) 2023-2025 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/mdspan.hpp>
#include <complex>
#include <concepts>
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

/// @private Constant of maximum compile time optimized block sizes.
constexpr int MaxOptimizedBlockSize = 3;

/// @private Concept defining a variadic compile time or runtime variable. T
/// indicates the type that is stored and V the value. Either V equals T, i.e.
/// it is a runtime variable or V defines a compile time value V::value of type
/// T.
template <typename T, typename V>
concept ConstexprType = std::is_same_v<T, V> || (requires {
                          typename V::value_type;
                          requires std::is_same_v<typename V::value_type, T>;
                        });

/// @private Check if ConstexprType holds a compile time constant.
template <typename T, typename V>
  requires ConstexprType<T, V>
constexpr bool is_compile_time_v = !std::is_same_v<T, V>;

/// @private Check if ConstexprType holds a run time variable.
template <typename T, typename V>
  requires ConstexprType<T, V>
constexpr bool is_runtime_v = std::is_same_v<T, V>;

template <typename T, typename V>
  requires ConstexprType<T, V>
T value(V container)
{
  if constexpr (is_compile_time_v<T, V>)
    return V::value;

  return container;
}

/// @private Concept capturing both compile time defined block sizes and runtime
/// ones.
template <typename V>
concept BlockSize = ConstexprType<int, V>;

/// @private Short notation for a compile time block size.
template <int N>
using BS = std::integral_constant<int, N>;

/// @private Check if block size is a compile time constant.
template <BlockSize V>
constexpr bool is_compile_time_bs_v = is_compile_time_v<int, V>;

/// @private Check if block size is a run time constant.
template <BlockSize V>
constexpr bool is_runtime_bs_v = is_runtime_v<int, V>;

/// @private Retrieves the integral block size of a runtime or compile time
/// block size.
int block_size(BlockSize auto bs) { return value<int, decltype(bs)>(bs); }

} // namespace dolfinx
