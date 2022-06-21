// Copyright (C) 2017 Chris Richardson, Garth N. Wells and Tormod Landet
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include <mpi4py/mpi4py.h>
#include <pybind11/pybind11.h>

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

namespace pybind11
{
namespace detail
{
template <>
class type_caster<dolfinx_wrappers::MPICommWrapper>
{
public:
  // Define this->value of type MPICommWrapper
  PYBIND11_TYPE_CASTER(dolfinx_wrappers::MPICommWrapper, _("MPICommWrapper"));

  // Python to C++
  bool load(handle src, bool)
  {
    // Check whether src is an mpi4py communicator
    VERIFY_MPI4PY(PyMPIComm_Get);
    if (PyObject_TypeCheck(src.ptr(), &PyMPIComm_Type))
    {
      value = dolfinx_wrappers::MPICommWrapper(*PyMPIComm_Get(src.ptr()));
      return true;
    }

    return false;
  }

  // C++ to Python
  static handle cast(dolfinx_wrappers::MPICommWrapper src,
                     pybind11::return_value_policy policy, handle parent)
  {
    VERIFY_MPI4PY(PyMPIComm_New);
    return pybind11::handle(PyMPIComm_New(src.get()));
  }

  operator dolfinx_wrappers::MPICommWrapper() { return this->value; }
};
} // namespace detail
} // namespace pybind11
