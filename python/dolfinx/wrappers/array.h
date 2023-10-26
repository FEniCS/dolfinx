// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{

/// Create an n-dimensional nb::ndarray that shares data with a
/// std::vector. The std::vector owns the data, and the nb::ndarray
/// object keeps the std::vector alive.
/// From https://github.com/pybind/pybind11/issues/1042

template <typename V, typename U>
auto as_nbndarray_new(V&& x, U&& shape)
{
  using _V = std::decay_t<V>;
  std::size_t dim = shape.size();
  typename _V::value_type* data = x.data();
  std::unique_ptr<_V> x_ptr = std::make_unique<_V>(std::move(x));
  auto capsule = nb::capsule(x_ptr.get(), [](void* p) noexcept
                             { delete reinterpret_cast<_V*>(p); });
  x_ptr.release();
  return nb::ndarray<nb::numpy, typename _V::value_type>(
      static_cast<typename _V::value_type*>(data), dim, shape.data(), capsule);
}

/// Create a nb::ndarray that shares data with a std::vector. The
/// std::vector owns the data, and the nb::ndarray object keeps the
/// std::vector alive.
// From https://github.com/pybind/pybind11/issues/1042
template <typename V>
auto as_nbndarray(V&& x)
{
  return as_nbndarray_new(std::move(x), std::array{x.size()});
}

template <typename V, typename U>
auto as_nbarray(V&& x, U&& shape)
{
  std::size_t dim = shape.size();
  auto data = x.data();
  std::unique_ptr<V> x_ptr = std::make_unique<V>(std::move(x));
  auto capsule = nb::capsule(x_ptr.get(), [](void* p) noexcept
                             { std::unique_ptr<V>(reinterpret_cast<V*>(p)); });
  x_ptr.release();
  return nb::ndarray<typename V::value_type, nb::numpy>(
      static_cast<typename V::value_type*>(data), dim, shape.data(), capsule);
}

/// Create a nb::ndarray that shares data with a std::vector. The
/// std::vector owns the data, and the nb::ndarray object keeps the std::vector
/// alive.
// From https://github.com/pybind/pybind11/issues/1042
template <typename V>
auto as_nbarray(V&& x)
{
  return as_nbarray(std::move(x), std::array{x.size()});
}

} // namespace dolfinx_wrappers
