// Copyright (C) 2017-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mpi_wrappers.h"
#include <cstdint>
#include <dolfinx/common/Scatterer.h>
#include <format>
#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <span>
#include <stdexcept>
#include <string_view>

namespace dolfinx_wrappers
{

/// @brief Check that a packed scatter buffer is large enough.
///
/// The scatter index arrays hold *block* indices packed by position,
/// so a buffer used with them holds `bs` values for each entry in
/// `idx`, regardless of the index values themselves.
///
/// @param[in] idx Block indices used to pack/unpack the buffer.
/// @param[in] size Number of entries in the buffer.
/// @param[in] bs Number of entries per block.
/// @param[in] name Name of the buffer, used in the error message.
inline void check_scatter_buffer(std::span<const std::int32_t> idx,
                                 std::size_t size, int bs,
                                 std::string_view name)
{
  std::size_t required = idx.size() * bs;
  if (size < required)
  {
    throw std::invalid_argument(
        std::format("{} buffer is too small: it has {} entries, but needs {}.",
                    name, size, required));
  }
}

/// @brief Bind the `scatter_fwd_begin`/`scatter_fwd_end`/`scatter_rev_begin`/
/// `scatter_rev_end` methods of `common::Scatterer` for a given scalar type.
///
/// These are thin wrappers around the equivalent `common::Scatterer`
/// methods: the caller packs/unpacks the send/receive buffers (sized
/// `bs * local_indices_block().size()` and
/// `bs * remote_indices_block().size()`) and must keep them alive
/// between a `*_begin` call and the matching `*_end` call.
template <typename T>
void declare_scatter_functions(
    nanobind::class_<dolfinx::common::Scatterer<>>& sc)
{
  namespace nb = nanobind;

  sc.def(
      "scatter_fwd_begin",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> local_buffer,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> remote_buffer,
         int bs) -> MPIRequestWrapper
      {
        check_scatter_buffer(self.local_indices_block(), local_buffer.size(),
                             bs, "Local packing");
        check_scatter_buffer(self.remote_indices_block(), remote_buffer.size(),
                             bs, "Remote packing");

        MPI_Request request = MPI_REQUEST_NULL;
        self.scatter_fwd_begin(local_buffer.data(), remote_buffer.data(), bs,
                               request);
        return MPIRequestWrapper(request);
      },
      nb::arg("local_buffer"), nb::arg("remote_buffer"), nb::arg("bs"));

  sc.def(
      "scatter_fwd_end",
      [](dolfinx::common::Scatterer<>& self, MPIRequestWrapper request)
      {
        MPI_Request req = request.get();
        self.scatter_fwd_end(req);
      },
      nb::arg("request"));

  sc.def(
      "scatter_rev_begin",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> remote_buffer,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> local_buffer,
         int bs) -> MPIRequestWrapper
      {
        check_scatter_buffer(self.remote_indices_block(), remote_buffer.size(),
                             bs, "Remote packing");
        check_scatter_buffer(self.local_indices_block(), local_buffer.size(),
                             bs, "Local packing");

        MPI_Request request = MPI_REQUEST_NULL;
        self.scatter_rev_begin<T>(remote_buffer.data(), local_buffer.data(), bs,
                                  request);
        return MPIRequestWrapper(request);
      },
      nb::arg("remote_buffer"), nb::arg("local_buffer"), nb::arg("bs"));

  sc.def(
      "scatter_rev_end",
      [](dolfinx::common::Scatterer<>& self, MPIRequestWrapper request)
      {
        MPI_Request req = request.get();
        self.scatter_rev_end(req);
      },
      nb::arg("request"));
}

} // namespace dolfinx_wrappers
