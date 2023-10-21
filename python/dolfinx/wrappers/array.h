// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

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
auto as_nbndarray_new(V&& array, U&& shape)
{
  using _V = std::decay_t<V>;
  std::size_t dim = shape.size();
  typename _V::value_type* data = array.data();
  std::unique_ptr<_V> x_ptr = std::make_unique<_V>(std::move(array));
  auto capsule
      = nb::capsule(x_ptr.get(), [](void* p) noexcept
                    { std::unique_ptr<_V>(reinterpret_cast<_V*>(p)); });
  x_ptr.release();
  return nb::ndarray<nb::numpy, typename _V::value_type>(
      static_cast<typename _V::value_type*>(data), dim, shape.data(), capsule);
}

template <typename Sequence, typename U>
auto as_nbarray(Sequence&& seq, U&& shape)
{
  std::size_t dim = shape.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr
      = std::make_unique<Sequence>(std::move(seq));
  auto capsule = nb::capsule(
      seq_ptr.get(), [](void* p) noexcept
      { std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p)); });
  seq_ptr.release();

  return nb::ndarray<typename Sequence::value_type, nb::numpy>(
      static_cast<typename Sequence::value_type*>(data), dim, shape.data(),
      capsule);
}

/// Create a nb::ndarray that shares data with a std::vector. The
/// std::vector owns the data, and the nb::ndarray object keeps the std::vector
/// alive.
// From https://github.com/pybind/pybind11/issues/1042
template <typename Sequence>
auto as_nbarray(Sequence&& seq)
{
  return as_nbarray(std::move(seq), std::array{seq.size()});
}

} // namespace dolfinx_wrappers
