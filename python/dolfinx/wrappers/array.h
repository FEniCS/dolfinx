// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace dolfinx_wrappers
{

/// Create an n-dimensional py::array_t that shares data with a
/// std::vector. The std::vector owns the data, and the py::array_t
/// object keeps the std::vector alive.
/// From https://github.com/pybind/pybind11/issues/1042
template <typename Sequence, typename U>
py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq, U&& shape)
{
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr
      = std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(
      seq_ptr.get(), [](void* p)
      { std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p)); });
  seq_ptr.release();
  return py::array(shape, data, capsule);
}

/// Create a py::array_t that shares data with a std::vector. The
/// std::vector owns the data, and the py::array_t object keeps the std::vector
/// alive.
// From https://github.com/pybind/pybind11/issues/1042
template <typename Sequence>
py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq)
{
  return as_pyarray(std::move(seq), std::array{seq.size()});
}

/// Create a py::array_t that shares data with an
/// xtensor array. The C++ object owns the data, and the
/// py::array_t object keeps the C++ object alive.
// From https://github.com/pybind/pybind11/issues/1042
template <typename U>
auto xt_as_pyarray(U&& x)
{
  auto shape = x.shape();
  auto data = x.data();
  auto strides = x.strides();
  std::transform(strides.begin(), strides.end(), strides.begin(),
                 [](auto s) { return s * sizeof(typename U::value_type); });
  std::unique_ptr<U> x_ptr = std::make_unique<U>(std::move(x));
  auto capsule = py::capsule(x_ptr.get(), [](void* p)
                             { std::unique_ptr<U>(reinterpret_cast<U*>(p)); });
  x_ptr.release();

  return py::array_t<typename U::value_type>(shape, strides, data, capsule);
}
} // namespace dolfinx_wrappers
