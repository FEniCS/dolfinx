// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <type_traits>

namespace dolfinx::common
{
/// @private Concept defining a variadic compile time or runtime variable. T
/// indicates the type that is stored and V the value. Either V equals T, i.e.
/// it is a runtime variable or V defines a compile time value V::value of type
/// T.
/// @tparam T type of the value to be stored.
/// @tparam V container type. Usually T for a runtime variable or a
/// std::integral_constant<T, ...> for a compile time constant.
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

/// @private Retrieve value of a compile time constant form a ConstexprType.
template <typename T, typename V>
  requires ConstexprType<T, V>
consteval T value(V /* container */,
                  typename std::enable_if_t<is_compile_time_v<T, V>>* = 0)
{
  return V::value;
}

/// @private Retrieve value of runtime variable form a ConstexprType.
template <typename T, typename V>
  requires ConstexprType<T, V>
T value(V container, typename std::enable_if_t<is_runtime_v<T, V>>* = 0)
{
  return container;
}

} // namespace dolfinx::common
