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

#ifndef _DOLFIN_PYBIND11_PETSC
#define _DOLFIN_PYBIND11_PETSC

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef HAS_PETSC
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscdm.h>

#ifdef HAS_PYBIND11_PETSC4PY
#include <petsc4py/petsc4py.h>
#endif


// pybind11 casters for PETSc/petsc4py objects

// FIXME: Create a macro for casting code

namespace pybind11
{
  namespace detail
  {
    template <> class type_caster<_p_Vec>
    {
    public:
      PYBIND11_TYPE_CASTER(Vec, _("vec"));

      // Pass communicator from Python to C++
      bool load(handle src, bool)
      {
        // FIXME: check reference counting
        //std::cout << "Py to c++" << std::endl;
        #ifdef HAS_PYBIND11_PETSC4PY
        value = PyPetscVec_Get(src.ptr());
        return true;
        #else
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py");
        return false;
        #endif
      }

      // Cast from C++ to Python (cast to pointer)
      static handle cast(Vec src, pybind11::return_value_policy policy, handle parent)
      {
        // FIXME: check reference counting
        #ifdef HAS_PYBIND11_PETSC4PY
        std::cout << "C++ to Python" << std::endl;
        return pybind11::handle(PyPetscVec_New(src));
        #else
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py");
        return handle();
        #endif
      }

      operator Vec()
      { return value; }
    };

    template <> class type_caster<_p_Mat>
    {
    public:
      PYBIND11_TYPE_CASTER(Mat, _("mat"));

      // Pass communicator from Python to C++
      bool load(handle src, bool)
      {
        // FIXME: check reference counting
        //std::cout << "Py to c++" << std::endl;
        #ifdef HAS_PYBIND11_PETSC4PY
        value = PyPetscMat_Get(src.ptr());
        return true;
        #else
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py");
        return false;
        #endif
      }

      // Cast from C++ to Python (cast to pointer)
      static handle cast(Mat src, pybind11::return_value_policy policy, handle parent)
      {
        // FIXME: check reference counting
        #ifdef HAS_PYBIND11_PETSC4PY
        std::cout << "C++ to Python" << std::endl;
        return pybind11::handle(PyPetscMat_New(src));
        #else
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py");
        return handle();
        #endif
      }

      operator Mat()
      { return value; }
    };

    template <> class type_caster<_p_KSP>
    {
    public:
      PYBIND11_TYPE_CASTER(KSP, _("ksp"));

      // Pass communicator from Python to C++
      bool load(handle src, bool)
      {
        // FIXME: check reference counting
        //std::cout << "Py to c++" << std::endl;
        #ifdef HAS_PYBIND11_PETSC4PY
        value = PyPetscKSP_Get(src.ptr());
        return true;
        #else
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py");
        return false;
        #endif
      }

      // Cast from C++ to Python (cast to pointer)
      static handle cast(KSP src, pybind11::return_value_policy policy, handle parent)
      {
        // FIXME: check reference counting
        #ifdef HAS_PYBIND11_PETSC4PY
        //std::cout << "C++ to Python" << std::endl;
        return pybind11::handle(PyPetscKSP_New(src));
        #else
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py");
        return handle();
        #endif
      }

      operator KSP()
      { return value; }
    };

    template <> class type_caster<_p_DM>
    {
    public:
      PYBIND11_TYPE_CASTER(DM, _("DM"));

      // Pass communicator from Python to C++
      bool load(handle src, bool)
      {
        // FIXME: check reference counting
        #ifdef HAS_PYBIND11_PETSC4PY
        //std::cout << "Py to C++ (DM)" << std::endl;
        value = PyPetscDM_Get(src.ptr());
        //std::cout << "Returning" << std::endl;
        return true;
        #else
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py");
        return false;
        #endif

      }

      // Cast from C++ to Python (cast to pointer)
      static handle cast(DM src, pybind11::return_value_policy policy, handle parent)
      {
        // FIXME: check reference counting
        #ifdef HAS_PYBIND11_PETSC4PY
        //std::cout << "C++ to Python (DM)" << std::endl;
        return pybind11::handle(PyPetscDM_New(src));
        #else
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py");
        return pybind11::handle();
        #endif
      }

      operator DM()
      { return value; }
    };
  }
}

#endif
#endif
