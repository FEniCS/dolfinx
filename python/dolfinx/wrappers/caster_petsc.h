// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/DolfinXException.h>
#include <petsc4py/petsc4py.h>
#include <petscdm.h>
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>
#include <pybind11/pybind11.h>

// pybind11 casters for PETSc/petsc4py objects

namespace py = pybind11;

// Import petsc4py on demand
#define VERIFY_PETSC4PY(func)                                                  \
  if (!func)                                                                   \
  {                                                                            \
    if (import_petsc4py() != 0)                                                \
      throw DolfinXException("Error when importing petsc4py");                 \
  }

// Macro for casting between PETSc and petsc4py objects
#define PETSC_CASTER_MACRO(TYPE, P4PYTYPE, NAME)                               \
  template <>                                                                  \
  class type_caster<_p_##TYPE>                                                 \
  {                                                                            \
  public:                                                                      \
    PYBIND11_TYPE_CASTER(TYPE, _(#NAME));                                      \
    bool load(handle src, bool)                                                \
    {                                                                          \
      VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_Get);                                \
      if (PyObject_TypeCheck(src.ptr(), &PyPetsc##P4PYTYPE##_Type) != 0)       \
      {                                                                        \
        value = PyPetsc##P4PYTYPE##_Get(src.ptr());                            \
        return true;                                                           \
      }                                                                        \
                                                                               \
      return false;                                                            \
    }                                                                          \
                                                                               \
    static handle cast(TYPE src, py::return_value_policy policy,               \
                       handle parent)                                          \
    {                                                                          \
      VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_New);                                \
      auto obj = PyPetsc##P4PYTYPE##_New(src);                                 \
      if (policy == py::return_value_policy::take_ownership)                   \
        PetscObjectDereference((PetscObject)src);                              \
      else if (policy == py::return_value_policy::reference_internal)          \
        keep_alive_impl(obj, parent);                                          \
      return py::handle(obj);                                                  \
    }                                                                          \
                                                                               \
    operator TYPE() { return value; }                                          \
  }

namespace pybind11::detail
{
PETSC_CASTER_MACRO(DM, DM, dm);
PETSC_CASTER_MACRO(IS, IS, is);
PETSC_CASTER_MACRO(KSP, KSP, ksp);
PETSC_CASTER_MACRO(Mat, Mat, mat);
PETSC_CASTER_MACRO(SNES, SNES, snes);
PETSC_CASTER_MACRO(Vec, Vec, vec);
} // namespace pybind11::detail

#undef PETSC_CASTER_MACRO
