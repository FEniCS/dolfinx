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
#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>

// pybind11 casters for PETSc/petsc4py objects
#ifdef HAS_PYBIND11_PETSC4PY
#include <petsc4py/petsc4py.h>

// Import petsc4py on demand
#define VERIFY_PETSC4PY(func)   \
  if (!func)                    \
  {                             \
    if (import_petsc4py() != 0) \
    {                           \
      std::cout << "ERROR: could not import petsc4py!" << std::endl; \
      throw std::runtime_error("Error when importing petsc4py");     \
    }                           \
  }

// Macro for casting between dolfin and petsc4py objects
#define PETSC_CASTER_MACRO(TYPE, NAME)          \
  template <> class type_caster<_p_##TYPE>      \
    {                                           \
    public:                                     \
      PYBIND11_TYPE_CASTER(TYPE, _(#NAME));     \
      bool load(handle src, bool)               \
      {                                         \
        VERIFY_PETSC4PY(PyPetsc##TYPE##_Get);   \
        if (PyObject_TypeCheck(src.ptr(), &PyPetsc##TYPE##_Type) == 0)  \
          return false;                                                 \
        value = PyPetsc##TYPE##_Get(src.ptr());                         \
        return true;                                                    \
      }                                                                 \
                                                                        \
      static handle cast(TYPE src, pybind11::return_value_policy policy, handle parent) \
      {                                                                 \
        VERIFY_PETSC4PY(PyPetsc##TYPE##_New);                           \
        return pybind11::handle(PyPetsc##TYPE##_New(src));              \
      }                                                                 \
                                                                        \
      operator TYPE()                                                   \
      { return value; }                                                 \
    }
#else
#define PETSC_CASTER_MACRO(TYPE, NAME)          \
  template <> class type_caster<_p_##TYPE>      \
    {                                           \
    public:                                     \
      PYBIND11_TYPE_CASTER(TYPE, _(#NAME));     \
      bool load(handle src, bool)               \
      {                                         \
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py"); \
        return false;                                                   \
      }                                                                 \
                                                                        \
      static handle cast(TYPE src, pybind11::return_value_policy policy, handle parent) \
      {                                                                 \
        throw std::runtime_error("DOLFIN has not been configured with petsc4py. Accessing underlying PETSc object requires petsc4py"); \
        return handle();                                                \
      }                                                                 \
                                                                        \
      operator TYPE()                                                   \
      { return value; }                                                 \
    }
#endif


namespace pybind11
{
  namespace detail
  {
    PETSC_CASTER_MACRO(DM, dm);
    PETSC_CASTER_MACRO(KSP, ksp);
    PETSC_CASTER_MACRO(Mat, mat);
    PETSC_CASTER_MACRO(SNES, snes);
    PETSC_CASTER_MACRO(Vec, vec);
  }
}

#undef PETSC_CASTER_MACRO

#endif
#endif
