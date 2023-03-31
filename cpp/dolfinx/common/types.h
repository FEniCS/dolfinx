// Copyright (C) 2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <type_traits>

namespace dolfinx
{
/// @private These structs are used to get the float/value type from a
/// template argument, including support for complex types.
template <typename T, typename = void>
struct scalar_value_type
{
  /// @internal
  typedef T value_type;
};
/// @private
template <typename T>
struct scalar_value_type<T, std::void_t<typename T::value_type>>
{
  typedef typename T::value_type value_type;
};
/// @private Convenience typedef
template <typename T>
using scalar_value_type_t = typename scalar_value_type<T>::value_type;
} // namespace dolfinx
