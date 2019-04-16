// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <petsc4py/petsc4py.h>
#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// pybind11 casters for PETSc/petsc4py objects

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

// Macro for casting between PETSc and petsc4py objects
#define PETSC_CASTER_MACRO(TYPE, P4PYTYPE, NAME)                                         \
  template <>                                                                  \
  class type_caster<_p_##TYPE>                                                 \
  {                                                                            \
  public:                                                                      \
    PYBIND11_TYPE_CASTER(TYPE, _(#NAME));                                      \
    bool load(handle src, bool)                                                \
    {                                                                          \
      if (src.is_none())                                                       \
      {                                                                        \
        value = nullptr;                                                       \
        return true;                                                           \
      }                                                                        \
      VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_Get);                                \
      if (PyObject_TypeCheck(src.ptr(), &PyPetsc##P4PYTYPE##_Type) == 0)       \
        return false;                                                          \
      value = PyPetsc##P4PYTYPE##_Get(src.ptr());                              \
      return true;                                                             \
    }                                                                          \
                                                                               \
    static handle cast(TYPE src, pybind11::return_value_policy policy,         \
                       handle parent)                                          \
    {                                                                          \
      VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_New);                                \
      auto obj = PyPetsc##P4PYTYPE##_New(src);                                 \
      if (policy == pybind11::return_value_policy::take_ownership)             \
        PetscObjectDereference((PetscObject)src);                              \
      return pybind11::handle(obj);                                            \
    }                                                                          \
                                                                               \
    operator TYPE() { return value; }                                          \
  }

namespace pybind11
{
namespace detail
{
PETSC_CASTER_MACRO(DM, DM, dm);
PETSC_CASTER_MACRO(KSP, KSP, ksp);
PETSC_CASTER_MACRO(Mat, Mat, mat);
// PETSC_CASTER_MACRO(MatNullSpace, NullSpace, matnullspace);
PETSC_CASTER_MACRO(SNES, SNES, snes);
PETSC_CASTER_MACRO(Vec, Vec, vec);
} // namespace detail
} // namespace pybind11

#undef PETSC_CASTER_MACRO
