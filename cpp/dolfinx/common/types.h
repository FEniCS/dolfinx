// Copyright (C) 2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <complex>
#include <concepts>
#include <type_traits>

namespace dolfinx
{
/// @private This concept is used to constrain the a template type to floating
/// point real or complex types. Note that this concept is different to
/// std::floating_point which does not include std::complex.
template <class T>
concept scalar = std::is_floating_point_v<T>
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
using scalar_value_type_t = typename scalar_value<T>::type;
} // namespace dolfinx
