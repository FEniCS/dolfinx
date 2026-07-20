// Copyright (C) 2017 Chris Richardson, Garth N. Wells and Tormod Landet
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include <mpi4py/mpi4py.h>
#include <nanobind/nanobind.h>

namespace nanobind::detail
{
template <>
class type_caster<dolfinx_wrappers::MPICommWrapper>
{
public:
  // Define this->value of type MPICommWrapper
  NB_TYPE_CASTER(dolfinx_wrappers::MPICommWrapper, const_name("MPICommWrapper"))

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
} // namespace nanobind::detail
