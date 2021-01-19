// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{

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

template <typename Sequence>
inline py::array_t<typename Sequence::value_type>
to_pyarray(const Sequence& seq)
{
  return py::array(seq.size(), seq.data());
}

} // namespace dolfinx_wrappers
