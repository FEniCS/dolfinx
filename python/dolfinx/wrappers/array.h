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

template <typename Sequence, typename U>
nb::ndarray<typename Sequence::value_type, nb::numpy> as_nbarray(Sequence&& seq,
                                                                 U&& shape)
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
      data, dim, shape.data(), capsule);
}

/// Create a nb::ndarray that shares data with a std::vector. The
/// std::vector owns the data, and the nb::ndarray object keeps the std::vector
/// alive.
// From https://github.com/pybind/pybind11/issues/1042
template <typename Sequence>
nb::ndarray<typename Sequence::value_type, nb::numpy> as_nbarray(Sequence&& seq)
{
  return as_nbarray(std::move(seq), std::array{seq.size()});
}

} // namespace dolfinx_wrappers
