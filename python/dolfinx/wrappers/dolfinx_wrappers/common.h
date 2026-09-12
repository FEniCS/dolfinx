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
#include <vector>

namespace dolfinx_wrappers
{
namespace nb = nanobind;

/// @brief Check that a data array is large enough to be indexed by every
/// entry of a scatter index array.
///
/// The scatter index arrays hold *block* indices into the data array, so
/// entry `idx[i]` accesses `idx[i] * bs` to `idx[i] * bs + bs - 1` and
/// the array must hold at least `(max(idx) + 1) * bs` entries.
/// `idx.size()` counts the blocks taking part in the exchange and is
/// unrelated to the required extent of the array.
///
/// @param[in] idx Block indices used to pack/unpack the data array.
/// @param[in] size Number of entries in the data array.
/// @param[in] bs Number of entries per block.
/// @param[in] name Name of the data array, used in the error message.
inline void check_scatter_buffer(std::span<const std::int32_t> idx,
                                 std::size_t size, int bs,
                                 std::string_view name)
{
  auto it = std::ranges::max_element(idx);
  if (it == idx.end())
    return;

  std::size_t required = (static_cast<std::size_t>(*it) + 1) * bs;
  if (size < required)
  {
    throw std::invalid_argument(
        std::format("{} buffer is too small: it has {} entries, but block "
                    "index {} with block size {} needs {}.",
                    name, size, *it, bs, required));
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
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> remote_data, int bs)
      {
        check_scatter_buffer(self.local_indices_block(), local_data.size(), bs,
                             "Local data");
        check_scatter_buffer(self.remote_indices_block(), remote_data.size(),
                             bs, "Ghost data");

        std::vector<T> send_buffer(bs * self.local_indices_block().size());
        {
          auto _local_data = local_data.view();
          auto& idx = self.local_indices_block();
          for (std::size_t i = 0; i < idx.size(); ++i)
            for (int j = 0; j < bs; ++j)
              send_buffer[i * bs + j] = _local_data(idx[i] * bs + j);
        }
        std::vector<T> recv_buffer(bs * self.remote_indices_block().size());
        MPI_Request request = MPI_REQUEST_NULL;
        self.scatter_fwd_begin<T>(send_buffer.data(), recv_buffer.data(), bs,
                                  request);
        self.scatter_fwd_end(request);
        {
          auto _remote_data = remote_data.view();
          auto& idx = self.remote_indices_block();
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
        check_scatter_buffer(self.local_indices_block(), local_data.size(), bs,
                             "Local data");
        check_scatter_buffer(self.remote_indices_block(), remote_data.size(),
                             bs, "Ghost data");

        std::vector<T> send_buffer(bs * self.remote_indices_block().size());
        {
          auto _remote_data = remote_data.view();
          auto& idx = self.remote_indices_block();
          for (std::size_t i = 0; i < idx.size(); ++i)
            for (int j = 0; j < bs; ++j)
              send_buffer[i * bs + j] = _remote_data(idx[i] * bs + j);
        }
        std::vector<T> recv_buffer(bs * self.local_indices_block().size());
        MPI_Request request = MPI_REQUEST_NULL;
        self.scatter_rev_begin<T>(send_buffer.data(), recv_buffer.data(), bs,
                                  request);
        self.scatter_rev_end(request);
        {
          auto _local_data = local_data.view();
          auto& idx = self.local_indices_block();
          for (std::size_t i = 0; i < idx.size(); ++i)
            for (int j = 0; j < bs; ++j)
              _local_data(idx[i] * bs + j) += recv_buffer[i * bs + j];
        }
      },
      nb::arg("local_data"), nb::arg("remote_data"), nb::arg("bs"));
}

} // namespace dolfinx_wrappers
