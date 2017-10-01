// Copyright (C) 2017 Chris Richardson and Garth N. Wells
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

#ifdef HAS_MPI
#include <mpi.h>


namespace dolfin_wrappers
{
  class MPICommunicatorWrapper {
    MPI_Comm comm;
  public:
    MPICommunicatorWrapper() {};
    MPICommunicatorWrapper(MPI_Comm comm) { this->comm = comm; }
    MPICommunicatorWrapper& operator=(const MPI_Comm comm) { this->comm = comm; }
    MPI_Comm get() const { return comm; }
  };
}


#ifdef HAS_MPI4PY

#include <mpi4py/mpi4py.h>
 
// Macro for casting between dolfin and mpi4py objects

namespace pybind11
{
  namespace detail
  {
    using CommWrap = dolfin_wrappers::MPICommunicatorWrapper

    template <> class type_caster<CommWrap>
      {
      public:
        // define this->value of type CommWrap
        PYBIND11_TYPE_CASTER(CommWrap, _("mpi_communicator"));

        // Python to C++
        bool load(handle src, bool)
        {
          value = CommWrap(PyMPIComm_Get(src.ptr()));
          return true;
        }

        // C++ to Python
        static handle cast(CommWrap src, pybind11::return_value_policy policy, handle parent)
        {
          return pybind11::handle(PyMPIComm_New(src.get()));
        }

        operator CommWrap()
        { return value; }
    };
  }
}

#endif // HAS_MPI4PY
#endif // HAS_MPI
#endif
