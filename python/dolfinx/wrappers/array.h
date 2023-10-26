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

/// Create an n-dimensional nb::ndarray that shares data with a
/// std::vector. The std::vector owns the data, and the nb::ndarray
/// object keeps the std::vector alive.
/// From https://github.com/pybind/pybind11/issues/1042
template <typename V>
auto as_nbarray(V&& x, std::size_t ndim, const std::size_t* shape)
{
  using _V = std::decay_t<V>;
  auto data = x.data();
  std::unique_ptr<_V> ptr = std::make_unique<_V>(std::move(x));
  // auto capsule = nb::capsule(x_ptr.get(), [](void* p) noexcept
  //                            { delete reinterpret_cast<_V*>(p); });
  auto capsule
      = nb::capsule(ptr.get(), [](void* p) noexcept
                    { std::unique_ptr<_V>(reinterpret_cast<_V*>(p)); });
  ptr.release();
  return nb::ndarray<typename _V::value_type, nb::numpy>(
      static_cast<typename _V::value_type*>(data), ndim, shape, capsule);
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
