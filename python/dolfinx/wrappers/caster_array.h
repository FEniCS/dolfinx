// Copyright (C) 2021 Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/array_2d.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pybind11::detail
{
template <typename T>
struct type_caster<dolfinx::common::array_2d<T>>
{
public:
  PYBIND11_TYPE_CASTER(dolfinx::common::array_2d<T>,
                       _("dolfinx::common::array_2d<T>"));

  // Conversion part 1 (Python -> C++)
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
      shape[i] = buf.shape()[i];

    value = dolfinx::common::array_2d<T>(shape, buf.data(),
                                         buf.data() + buf.size());

    return true;
  }

  // Conversion part 2 (C++ -> Python)
  static py::handle cast(const dolfinx::common::array_2d<T>& src,
                         py::return_value_policy policy, py::handle parent)
  {

    std::vector<std::size_t> shape(2);
    std::vector<std::size_t> strides(2, 0);

    shape[0] = src.rows();
    shape[1] = src.cols();
    strides[0] = src.cols();

    py::array a(std::move(shape), std::move(strides), src.data());

    return a.release();
  }
};
} // namespace pybind11::detail