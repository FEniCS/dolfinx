// Copyright (C) 2021-2025 Garth N. Wells
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

namespace nb = nanobind;

namespace dolfinx_wrappers
{
/// @brief Create a multi-dimensional `nb::ndarray` that shares data
/// with a `std::vector`.
///
/// The `std::vector` owns the data, and the `nb::ndarray` object keeps
/// the `std::vector` alive. Layout is row-major.
///
/// @tparam V `std::vector` type.
/// @param x `std::vector` to move into the `nb::ndarray`.
/// @param ndim Number of array dimensions (rank).
/// @param shape Shape of the array.
/// @return An n-dimensional array that shares data with `x`.
template <typename V>
  requires std::movable<V>
auto as_nbarray(V&& x, std::size_t ndim, const std::size_t* shape)
{
  using _V = std::decay_t<V>;
  _V* ptr = new _V(std::forward<V>(x));
  return nb::ndarray<typename _V::value_type, nb::numpy>(
      ptr->data(), ndim, shape,
      nb::capsule(ptr, [](void* p) noexcept { delete (_V*)p; }));
}

/// @brief Create a multi-dimensional `nb::ndarray` that shares data
/// with a `std::vector`.
///
/// The `std::vector` owns the data, and the `nb::ndarray` object keeps
/// the `std::vector` alive. Layout is row-major.
///
/// @tparam V `std::vector` type.
/// @param x `std::vector` to move into the `nb::ndarray`.
/// @param shape Shape of the array.
/// @return An n-dimensional array that shares data with `x`.
template <typename V>
  requires std::movable<V>
auto as_nbarray(V&& x, std::initializer_list<std::size_t> shape)
{
  return as_nbarray(std::forward<V>(x), shape.size(), shape.begin());
}

/// @brief Create a multi-dimensional `nb::ndarray` that shares data
/// with a `std::vector`.
///
/// The `std::vector` owns the data, and the `nb::ndarray` object keeps
/// the `std::vector` alive. Layout is row-major.
///
/// @tparam V `std::vector` type.
/// @tparam W Shape container type.
/// @param x `std::vector` to move into the `nb::ndarray`.
/// @param shape Container that hold the shape of the array.
/// @return An n-dimensional array that shares data with `x`.
template <typename V, typename W>
  requires std::movable<V>
auto as_nbarray(V&& x, W&& shape)
{
  return as_nbarray(std::forward<V>(x), shape.size(), shape.begin());
}

/// @brief Create a 1D `nb::ndarray` that shares data with a
/// `std::vector`.
///
/// The std::vector owns the data, and the nb::ndarray object keeps the
/// std::vector alive.
///
/// @tparam V `std::vector` type.
/// @param x `std::vector` to move into the `nb::ndarray`.
/// @return An 1D array that shares data with `x`.
template <typename V>
  requires std::movable<V>
auto as_nbarray(V&& x)
{
  return as_nbarray(std::forward<V>(x), {x.size()});
}

} // namespace dolfinx_wrappers
