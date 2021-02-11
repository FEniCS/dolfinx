// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/array2d.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{

/// Create a py::array_t that shares data with a std::vector. The
/// std::vector owns the data, and the py::array_t object keeps the std::vector
/// alive.
// From https://github.com/pybind/pybind11/issues/1042
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq)
{
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr
      = std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void* p) {
    std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p));
  });
  seq_ptr.release();
  return py::array(size, data, capsule);
}

/// Create a py::array_t that shares data with a
/// dolfinx::common::array2d. The C++ array2d owns the data, and the
/// py::array_t object keeps the C++ object alive .
// From https://github.com/pybind/pybind11/issues/1042
template <typename T>
inline py::array_t<T> as_pyarray2d(dolfinx::common::array2d<T>&& array)
{
  auto shape = array.shape;
  auto strides = array.strides();
  auto data = array.data();
  std::unique_ptr<dolfinx::common::array2d<T>> array_ptr
      = std::make_unique<dolfinx::common::array2d<T>>(std::move(array));
  auto capsule = py::capsule(array_ptr.get(), [](void* p) {
    std::unique_ptr<dolfinx::common::array2d<T>>(
        reinterpret_cast<dolfinx::common::array2d<T>*>(p));
  });
  array_ptr.release();
  return py::array(shape, strides, data, capsule);
}

} // namespace dolfinx_wrappers
