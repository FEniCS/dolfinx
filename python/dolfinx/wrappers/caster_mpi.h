// Copyright (C) 2017 Chris Richardson, Garth N. Wells and Tormod Landet
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include <mpi4py/mpi4py.h>
#include <nanobind/nanobind.h>

// Import mpi4py on demand
#define VERIFY_MPI4PY(func)                                                    \
  if (!func)                                                                   \
  {                                                                            \
    int rc = import_mpi4py();                                                  \
    if (rc != 0)                                                               \
    {                                                                          \
      throw std::runtime_error("Error when importing mpi4py");                 \
    }                                                                          \
  }

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
    VERIFY_MPI4PY(PyMPIComm_Get);
    if (PyObject_TypeCheck(src.ptr(), &PyMPIComm_Type))
    {
      value = dolfinx_wrappers::MPICommWrapper(*PyMPIComm_Get(src.ptr()));
      return true;
    }
    else
      return false;
  }

  // C++ -> Python
  static handle from_cpp(dolfinx_wrappers::MPICommWrapper src, rv_policy policy,
                         cleanup_list* /*cleanup*/) noexcept
  {
    if (policy != rv_policy::automatic
        and policy != rv_policy::automatic_reference
        and policy != rv_policy::reference_internal)
    {
      return {};
    }
    VERIFY_MPI4PY(PyMPIComm_New);
    PyObject* c = PyMPIComm_New(src.get());
    return nanobind::handle(c);
  }

  operator dolfinx_wrappers::MPICommWrapper() { return this->value; }
};
} // namespace nanobind::detail
