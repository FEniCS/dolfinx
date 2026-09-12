// Copyright (C) 2017-2026 Chris Richardson, Garth N. Wells and Tormod Landet
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mpi_wrappers.h"
#include <mpi4py/mpi4py.h>
#include <nanobind/nanobind.h>

namespace nanobind::detail
{
template <>
class type_caster<dolfinx_wrappers::MPICommWrapper>
{
public:
  // Define this->value of type MPICommWrapper
  NB_TYPE_CASTER(dolfinx_wrappers::MPICommWrapper,
                 const_name("mpi4py.MPI.Comm"))

  // Python -> C++
  bool from_python(handle src, uint8_t /*flags*/,
                   cleanup_list* /*cleanup*/) noexcept
  {
    if (!PyMPIComm_Get)
    {
      if (import_mpi4py() != 0)
        return false;
    }

    if (PyObject_TypeCheck(src.ptr(), &PyMPIComm_Type))
    {
      value = dolfinx_wrappers::MPICommWrapper(*PyMPIComm_Get(src.ptr()));
      return true;
    }
    else
      return false;
  }

  // C++ -> Python
  static handle from_cpp(const dolfinx_wrappers::MPICommWrapper& src,
                         rv_policy policy, cleanup_list* /*cleanup*/) noexcept
  {
    // MPICommWrapper always wraps a plain communicator handle by
    // value, so every policy other than `none` (which must not
    // create a new object) behaves identically here.
    if (policy == rv_policy::none)
      return {};

    if (!PyMPIComm_New)
    {
      if (import_mpi4py() != 0)
        return {};
    }

    PyObject* c = PyMPIComm_New(src.get());
    return nanobind::handle(c);
  }

  operator dolfinx_wrappers::MPICommWrapper() { return this->value; }
};

template <>
class type_caster<dolfinx_wrappers::MPIRequestWrapper>
{
public:
  // Define this->value of type MPIRequestWrapper
  NB_TYPE_CASTER(dolfinx_wrappers::MPIRequestWrapper,
                 const_name("mpi4py.MPI.Request"))

  // Python -> C++
  bool from_python(handle src, uint8_t /*flags*/,
                   cleanup_list* /*cleanup*/) noexcept
  {
    if (!PyMPIRequest_Get)
    {
      if (import_mpi4py() != 0)
        return false;
    }

    if (PyObject_TypeCheck(src.ptr(), &PyMPIRequest_Type))
    {
      value = dolfinx_wrappers::MPIRequestWrapper(*PyMPIRequest_Get(src.ptr()));
      return true;
    }
    else
      return false;
  }

  // C++ -> Python
  static handle from_cpp(const dolfinx_wrappers::MPIRequestWrapper& src,
                         rv_policy policy, cleanup_list* /*cleanup*/) noexcept
  {
    // MPIRequestWrapper always wraps a plain request handle by value,
    // so every policy other than `none` (which must not create a new
    // object) behaves identically here.
    if (policy == rv_policy::none)
      return {};

    if (!PyMPIRequest_New)
    {
      if (import_mpi4py() != 0)
        return {};
    }

    PyObject* r = PyMPIRequest_New(src.get());
    return nanobind::handle(r);
  }

  operator dolfinx_wrappers::MPIRequestWrapper() { return this->value; }
};
} // namespace nanobind::detail
