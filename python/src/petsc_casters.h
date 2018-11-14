// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>

// pybind11 casters for PETSc/petsc4py objects
#include <petsc4py/petsc4py.h>

// Import petsc4py on demand
#define VERIFY_PETSC4PY(func)                                                  \
  if (!func)                                                                   \
  {                                                                            \
    if (import_petsc4py() != 0)                                                \
    {                                                                          \
      std::cout << "ERROR: could not import petsc4py!" << std::endl;           \
      throw std::runtime_error("Error when importing petsc4py");               \
    }                                                                          \
  }

// Macro for casting between dolfin and petsc4py objects
#define PETSC_CASTER_MACRO(TYPE, NAME)                                         \
  template <>                                                                  \
  class type_caster<_p_##TYPE>                                                 \
  {                                                                            \
  public:                                                                      \
    PYBIND11_TYPE_CASTER(TYPE, _(#NAME));                                      \
    bool load(handle src, bool)                                                \
    {                                                                          \
      VERIFY_PETSC4PY(PyPetsc##TYPE##_Get);                                    \
      if (PyObject_TypeCheck(src.ptr(), &PyPetsc##TYPE##_Type) == 0)           \
        return false;                                                          \
      value = PyPetsc##TYPE##_Get(src.ptr());                                  \
      return true;                                                             \
    }                                                                          \
                                                                               \
    static handle cast(TYPE src, pybind11::return_value_policy policy,         \
                       handle parent)                                          \
    {                                                                          \
      VERIFY_PETSC4PY(PyPetsc##TYPE##_New);                                    \
      return pybind11::handle(PyPetsc##TYPE##_New(src));                       \
    }                                                                          \
                                                                               \
    operator TYPE() { return value; }                                          \
  }

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
