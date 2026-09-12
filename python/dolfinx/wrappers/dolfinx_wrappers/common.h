// Copyright (C) 2017-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <dolfinx/common/Scatterer.h>
#include <format>
#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace dolfinx_wrappers
{
namespace nb = nanobind;

/// @brief Check that every entry of `idx` is a valid index into an array
/// of `size` entries.
///
/// The scatter index arrays index *into* the data array, so the array
/// must hold at least `max(idx) + 1` entries. `idx.size()` counts the
/// entries taking part in the exchange and is unrelated to the required
/// extent of the data array.
///
/// @param[in] idx Indices used to pack/unpack the data array.
/// @param[in] size Number of entries in the data array.
/// @param[in] name Name of the data array, used in the error message.
inline void check_scatter_buffer(std::span<const std::int32_t> idx,
                                 std::size_t size, std::string_view name)
{
  auto it = std::ranges::max_element(idx);
  if (it != idx.end() and std::cmp_greater_equal(*it, size))
  {
    throw std::invalid_argument(
        std::format("{} buffer is too small: it has {} entries, but index {} "
                    "is accessed.",
                    name, size, *it));
  }
}

template <typename T>
void declare_scatter_functions(
    nanobind::class_<dolfinx::common::Scatterer<>>& sc)
{
  sc.def(
      "scatter_fwd",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> local_data,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> remote_data)
      {
        check_scatter_buffer(self.local_indices(), local_data.size(),
                             "Local data");
        check_scatter_buffer(self.remote_indices(), remote_data.size(),
                             "Ghost data");

        std::vector<T> send_buffer(self.local_indices().size());
        {
          auto _local_data = local_data.view();
          auto& idx = self.local_indices();
          for (std::size_t i = 0; i < idx.size(); ++i)
            send_buffer[i] = _local_data(idx[i]);
        }
        std::vector<T> recv_buffer(self.remote_indices().size());
        MPI_Request request = MPI_REQUEST_NULL;
        self.scatter_fwd_begin<T>(send_buffer.data(), recv_buffer.data(),
                                  request);
        self.scatter_fwd_end(request);
        {
          auto _remote_data = remote_data.view();
          auto& idx = self.remote_indices();
          for (std::size_t i = 0; i < idx.size(); ++i)
            _remote_data(idx[i]) = recv_buffer[i];
        }
      },
      nb::arg("local_data"), nb::arg("remote_data"));

  sc.def(
      "scatter_rev",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> local_data,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> remote_data)
      {
        check_scatter_buffer(self.local_indices(), local_data.size(),
                             "Local data");
        check_scatter_buffer(self.remote_indices(), remote_data.size(),
                             "Ghost data");

        std::vector<T> send_buffer(self.remote_indices().size());
        {
          auto _remote_data = remote_data.view();
          auto& idx = self.remote_indices();
          for (std::size_t i = 0; i < idx.size(); ++i)
            send_buffer[i] = _remote_data(idx[i]);
        }
        std::vector<T> recv_buffer(self.local_indices().size());
        MPI_Request request = MPI_REQUEST_NULL;
        self.scatter_rev_begin<T>(send_buffer.data(), recv_buffer.data(),
                                  request);
        self.scatter_rev_end(request);
        {
          auto _local_data = local_data.view();
          auto& idx = self.local_indices();
          for (std::size_t i = 0; i < idx.size(); ++i)
            _local_data(idx[i]) += recv_buffer[i];
        }
      },
      nb::arg("local_data"), nb::arg("remote_data"));
}

} // namespace dolfinx_wrappers
