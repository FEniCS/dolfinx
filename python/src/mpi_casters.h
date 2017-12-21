// Copyright (C) 2017 Chris Richardson, Garth N. Wells and Tormod Landet
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifndef _DOLFIN_PYBIND11_MPI
#define _DOLFIN_PYBIND11_MPI

#include <pybind11/pybind11.h>
#include "MPICommWrapper.h"
#include <dolfin/common/SubSystemsManager.h>

// Macro for casting between dolfin and mpi4py MPI communicators
#ifdef HAS_MPI
#ifdef HAS_PYBIND11_MPI4PY
#include <mpi4py/mpi4py.h>

// Import mpi4py on demand
#define VERIFY_MPI4PY(func)     \
  if (!func)                    \
  {                             \
    dolfin::SubSystemsManager::init_mpi();      \
    int rc = import_mpi4py();                     \
    if (rc != 0)   \
    {                           \
      std::cout << "ERROR: could not import mpi4py!" << std::endl; \
      throw std::runtime_error("Error when importing mpi4py");     \
    }                           \
  }

namespace pybind11
{
  namespace detail
  {
    template <> class type_caster<dolfin_wrappers::MPICommWrapper>
      {
      public:
        // Define this->value of type MPICommWrapper
        PYBIND11_TYPE_CASTER(dolfin_wrappers::MPICommWrapper, _("MPICommWrapper"));

        // Python to C++
        bool load(handle src, bool)
        {
          // Simplified version of isinstance(src, mpi4py.MPI.Comm) - avoids segfault
          // when pybind11 tries to convert some other random type to MPICommWrapper
          if (not hasattr(src, "Allgather"))
            return false;
          VERIFY_MPI4PY(PyMPIComm_Get);
          value = dolfin_wrappers::MPICommWrapper(*PyMPIComm_Get(src.ptr()));
          return true;
        }

        // C++ to Python
        static handle cast(dolfin_wrappers::MPICommWrapper src,
            pybind11::return_value_policy policy, handle parent)
        {
          VERIFY_MPI4PY(PyMPIComm_New);
          return pybind11::handle(PyMPIComm_New(src.get()));
        }

        operator dolfin_wrappers::MPICommWrapper()
        {
          return this->value;
        }
    };
  }
}

#endif // HAS_PYBIND11_MPI4PY
#endif // HAS_MPI
#endif
