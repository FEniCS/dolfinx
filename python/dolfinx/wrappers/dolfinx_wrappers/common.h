// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/Scatterer.h>
#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <vector>

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
        if (local_data.size() < bs * self.local_indices().size())
        {
          throw std::runtime_error(
              "Local data buffer too small in forward scatter.");
        }
        if (remote_data.size() < bs * self.remote_indices().size())
        {
          throw std::runtime_error(
              "Ghost data buffer too small in forward scatter.");
        }

        std::vector<T> send_buffer(bs * self.local_indices().size());
        {
          auto _local_data = local_data.view();
          auto& idx = self.local_indices();
          for (std::size_t i = 0; i < idx.size(); ++i)
            for (int j = 0; j < bs; ++j)
              send_buffer[i * bs + j] = _local_data(idx[i] * bs + j);
        }
        std::vector<T> recv_buffer(bs * self.remote_indices().size());
        MPI_Request request = MPI_REQUEST_NULL;
        self.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), bs,
                               request);
        self.scatter_fwd_end(request);
        {
          auto _remote_data = remote_data.view();
          auto& idx = self.remote_indices();
          for (std::size_t i = 0; i < idx.size(); ++i)
            for (int j = 0; j < bs; ++j)
              _remote_data(idx[i] * bs + j) = recv_buffer[i * bs + j];
        }
      },
      nb::arg("local_data"), nb::arg("remote_data"), nb::arg("bs"));

  sc.def(
      "scatter_rev",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> local_data,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> remote_data, int bs)
      {
        if (local_data.size() < bs * self.local_indices().size())
        {
          throw std::runtime_error(
              "Local data buffer too small in reverse scatter.");
        }
        if (remote_data.size() < bs * self.remote_indices().size())
        {
          throw std::runtime_error(
              "Ghost data buffer too small in reverse scatter.");
        }

        std::vector<T> send_buffer(bs * self.remote_indices().size());
        {
          auto _remote_data = remote_data.view();
          auto& idx = self.remote_indices();
          for (std::size_t i = 0; i < idx.size(); ++i)
            for (int j = 0; j < bs; ++j)
              send_buffer[i * bs + j] = _remote_data(idx[i] * bs + j);
        }
        std::vector<T> recv_buffer(bs * self.local_indices().size());
        MPI_Request request = MPI_REQUEST_NULL;
        self.scatter_rev_begin<T>(send_buffer.data(), recv_buffer.data(), bs,
                                  request);
        self.scatter_rev_end(request);
        {
          auto _local_data = local_data.view();
          auto& idx = self.local_indices();
          for (std::size_t i = 0; i < idx.size(); ++i)
            for (int j = 0; j < bs; ++j)
              _local_data(idx[i] * bs + j) += recv_buffer[i * bs + j];
        }
      },
      nb::arg("local_data"), nb::arg("remote_data"), nb::arg("bs"));
}

} // namespace dolfinx_wrappers
