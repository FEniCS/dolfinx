// Copyright (C) 2021-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <initializer_list>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
/// Create an n-dimensional nb::ndarray that shares data with a
/// std::vector.
///
/// The std::vector owns the data, and the nb::ndarray object keeps the
/// std::vector alive.
template <typename V>
  requires std::movable<V>
auto as_nbarray(V&& x, std::size_t ndim, const std::size_t* shape)
{
  using _V = std::decay_t<V>;
  _V* ptr = new _V(std::move(x));
  return nb::ndarray<typename _V::value_type, nb::numpy>(
      ptr->data(), ndim, shape,
      nb::capsule(ptr, [](void* p) noexcept { delete (_V*)p; }));
}

/// Create an n-dimensional nb::ndarray that shares data with a
/// std::vector.
///
/// The std::vector owns the data, and the nb::ndarray object keeps the
/// std::vector alive.
template <typename V>
  requires std::movable<V>
auto as_nbarray(V&& x, const std::initializer_list<std::size_t> shape)
{
  return as_nbarray(std::forward<V>(x), shape.size(), shape.begin());
}

/// Create a 1D nb::ndarray that shares data with a std::vector.
///
/// The std::vector owns the data, and the nb::ndarray object keeps the
/// std::vector alive.
template <typename V>
  requires std::movable<V>
auto as_nbarray(V&& x)
{
  return as_nbarray(std::forward<V>(x), {x.size()});
}

} // namespace dolfinx_wrappers
