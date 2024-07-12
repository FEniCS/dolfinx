// Copyright (C) 2021-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <initializer_list>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{

template <typename V>
auto as_nbarray_copy(const V& x, std::size_t ndim, const std::size_t* shape)
{
  using _V = std::decay_t<V>;
  using T = typename _V::value_type;
  T* ptr = new T[x.size()];
  std::ranges::copy(x, ptr);
  return nb::ndarray<T, nb::numpy>(
      ptr, ndim, shape,
      nb::capsule(ptr, [](void* p) noexcept { delete[] (T*)p; }));
}

template <typename V>
auto as_nbarray_copy(const V& x, const std::initializer_list<std::size_t> shape)
{
  return as_nbarray_copy(x, shape.size(), shape.begin());
}

/// Create an n-dimensional nb::ndarray that shares data with a
/// std::vector. The std::vector owns the data, and the nb::ndarray
/// object keeps the std::vector alive.
template <typename V>
auto as_nbarray(V&& x, std::size_t ndim, const std::size_t* shape)
{
  using _V = std::decay_t<V>;
  _V* ptr = new _V(std::move(x));
  return nb::ndarray<typename _V::value_type, nb::numpy>(
      ptr->data(), ndim, shape,
      nb::capsule(ptr, [](void* p) noexcept { delete (_V*)p; }));
}

template <typename V>
auto as_nbarray(V&& x, const std::initializer_list<std::size_t> shape)
{
  return as_nbarray(x, shape.size(), shape.begin());
}

/// Create a nb::ndarray that shares data with a std::vector. The
/// std::vector owns the data, and the nb::ndarray object keeps the
/// std::vector alive.
template <typename V>
auto as_nbarray(V&& x)
{
  return as_nbarray(std::move(x), {x.size()});
}

} // namespace dolfinx_wrappers
