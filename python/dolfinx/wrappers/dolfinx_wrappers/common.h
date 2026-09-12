// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/Scatterer.h>
#include <functional>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <span>
#include <stdexcept>

namespace dolfinx_wrappers
{

template <typename T>
void declare_scatter_functions(
    nanobind::class_<dolfinx::common::Scatterer<>>& sc)
{
  namespace nb = nanobind;

  sc.def(
      "scatter_fwd",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> local_data,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> remote_data, int bs)
      {
        if (local_data.size() < bs * self.local_indices_block().size())
        {
          throw std::runtime_error(
              "Local data buffer too small in forward scatter.");
        }
        if (remote_data.size() < bs * self.remote_indices_block().size())
        {
          throw std::runtime_error(
              "Ghost data buffer too small in forward scatter.");
        }

        dolfinx::common::scatter_fwd<T>(
            self, std::span<const T>(local_data.data(), local_data.size()),
            std::span<T>(remote_data.data(), remote_data.size()), bs);
      },
      nb::arg("local_data"), nb::arg("remote_data"), nb::arg("bs"));

  sc.def(
      "scatter_rev",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> local_data,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> remote_data, int bs)
      {
        if (local_data.size() < bs * self.local_indices_block().size())
        {
          throw std::runtime_error(
              "Local data buffer too small in reverse scatter.");
        }
        if (remote_data.size() < bs * self.remote_indices_block().size())
        {
          throw std::runtime_error(
              "Ghost data buffer too small in reverse scatter.");
        }

        dolfinx::common::scatter_rev<T>(
            self, std::span<T>(local_data.data(), local_data.size()),
            std::span<const T>(remote_data.data(), remote_data.size()), bs,
            std::plus<T>());
      },
      nb::arg("local_data"), nb::arg("remote_data"), nb::arg("bs"));
}

} // namespace dolfinx_wrappers
