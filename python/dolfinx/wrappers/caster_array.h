// Copyright (C) 2021 Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/array2d.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pybind11::detail
{
template <typename T>
struct type_caster<dolfinx::common::array2d<T>>
{
public:
  PYBIND11_TYPE_CASTER(dolfinx::common::array2d<T>,
                       _("dolfinx::common::array2d<T>"));

  // Conversion Python -> C++
  bool load(py::handle src, bool convert)
  {
    if (!convert and !py::array_t<T>::check_(src))
      return false;

    auto buf
        = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
            src);
    if (!buf)
      return false;

    auto dims = buf.ndim();
    if (dims != 2)
      return false;

    std::vector<size_t> shape(2);

    for (int i = 0; i < 2; ++i)
      shape[i] = buf.shape[i];

    value = dolfinx::common::array2d<T>(buf.shape(0), buf.shape(1));
    std::copy(buf.data(), buf.data() + buf.size(), value.data());

    return true;
  }

  /// Conversion C++ -> Python
  static py::handle cast(const dolfinx::common::array2d<T>& src,
                         py::return_value_policy policy, py::handle parent)
  {
    py::array a(src.shape, src.strides(), src.data());
    return a.release();
  }
};
} // namespace pybind11::detail