// Copyright (C) 2017-2026 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <format>
#include <functional>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <vector>

namespace dolfinx_wrappers
{

namespace nb = nanobind;

/// @brief Shape of a Python entity marker: candidate-entity coordinates
/// (a `(3, num_points)` array) mapped to a per-point boolean array.
template <typename T>
using PythonMarkerFunction
    = std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
        nb::ndarray<const T, nb::ndim<2>, nb::numpy>)>;

/// @brief Wrap a Python entity marker as the callable
/// dolfinx::mesh::locate_entities, dolfinx::mesh::locate_entities_boundary
/// and dolfinx::fem::locate_dofs_geometrical expect.
///
/// @note The returned closure holds a reference to `marker`, so it must
/// not outlive it. It is only ever passed straight into a DOLFINx call
/// that invokes it in place.
///
/// @param[in] marker Python marker function.
/// @return Callable mapping an mdspan of point coordinates to a marker
/// per point.
template <typename T>
auto to_cpp_marker(const PythonMarkerFunction<T>& marker)
{
  return [&marker](auto x)
  {
    nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(
        x.data_handle(), {x.extent(0), x.extent(1)});
    auto marked = marker(x_view);
    if (marked.size() != x.extent(1))
    {
      throw std::invalid_argument(
          std::format("Marker function returned {} values for {} points.",
                      marked.size(), x.extent(1)));
    }
    return std::vector<std::int8_t>(marked.data(),
                                    marked.data() + marked.size());
  };
}

} // namespace dolfinx_wrappers
