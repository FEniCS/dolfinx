// Copyright (C) 2017-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mpi_wrappers.h"
#include <dolfinx/common/Scatterer.h>
#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>

namespace dolfinx_wrappers
{

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
        if (local_buffer.size() < bs * self.local_indices_block().size())
          throw std::runtime_error("Local packing buffer too small.");
        if (remote_buffer.size() < bs * self.remote_indices_block().size())
          throw std::runtime_error("Remote packing buffer too small.");

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
        if (remote_buffer.size() < bs * self.remote_indices_block().size())
          throw std::runtime_error("Remote packing buffer too small.");
        if (local_buffer.size() < bs * self.local_indices_block().size())
          throw std::runtime_error("Local packing buffer too small.");

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
