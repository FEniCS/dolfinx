// Copyright (C) 2023-2025 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/mdspan.hpp>
#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace dolfinx
{
/// @private This concept is used to constrain the a template type to floating
/// point real or complex types. Note that this concept is different to
/// std::floating_point which does not include std::complex.

template <class T>
struct is_custom_scalar : std::false_type
{
};

template <class T>
concept scalar = std::floating_point<T>
                 || std::is_same_v<T, std::complex<typename T::value_type>>
                 || is_custom_scalar<T>::value;

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

/// @private Concept for a rank-2 mdspan-like type: exposes a
/// `value_type`, has static `rank() == 2`, and supports two-index access
/// and per-dimension `extent` queries.
template <typename T>
concept mdspan2 = requires(T x, std::size_t i) {
  typename T::value_type;
  requires T::rank() == 2;
  x(i, i);
  { x.extent(i) } -> std::integral;
};

} // namespace dolfinx
