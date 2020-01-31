// Copyright (C) 2017 Chris Richardson, Garth N. Wells and Tormod Landet
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include <dolfinx/common/SubSystemsManager.h>
#include <mpi4py/mpi4py.h>
#include <pybind11/pybind11.h>

// Import mpi4py on demand
#define VERIFY_MPI4PY(func)                                                    \
  if (!func)                                                                   \
  {                                                                            \
    dolfinx::common::SubSystemsManager::init_mpi();                             \
    int rc = import_mpi4py();                                                  \
    if (rc != 0)                                                               \
    {                                                                          \
      std::cout << "ERROR: could not import mpi4py!" << std::endl;             \
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
    // Simplified version of isinstance(src, mpi4py.MPI.Comm) - avoids
    // segfault when pybind11 tries to convert some other random type to
    // MPICommWrapper
    if (not hasattr(src, "Allgather"))
      return false;
    VERIFY_MPI4PY(PyMPIComm_Get);
    value = dolfinx_wrappers::MPICommWrapper(*PyMPIComm_Get(src.ptr()));
    return true;
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
