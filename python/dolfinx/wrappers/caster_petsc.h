// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <nanobind/nanobind.h>
#include <petsc4py/petsc4py.h>
#include <petscdm.h>
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>

// nanobind casters for PETSc/petsc4py objects

namespace nb = nanobind;

// Import petsc4py on demand
#define VERIFY_PETSC4PY(func)                                                  \
  if (!func)                                                                   \
  {                                                                            \
    if (import_petsc4py() != 0)                                                \
      throw std::runtime_error("Error when importing petsc4py");               \
  }

// Macro for casting between PETSc and petsc4py objects
#define PETSC_CASTER_MACRO(TYPE, P4PYTYPE, NAME)                               \
  template <>                                                                  \
  class type_caster<_p_##TYPE>                                                 \
  {                                                                            \
  public:                                                                      \
    NB_TYPE_CASTER(TYPE, const_name(#NAME));                                   \
    bool from_python(handle src, uint8_t, cleanup_list*)                       \
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
    static handle from_cpp(TYPE src, rv_policy policy, cleanup_list* cleanup)  \
    {                                                                          \
      VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_New);                                \
      auto obj = PyPetsc##P4PYTYPE##_New(src);                                 \
      if (policy == nb::rv_policy::take_ownership)                             \
        PetscObjectDereference((PetscObject)src);                              \
      else if (policy == nb::rv_policy::reference_internal)                    \
        nb::keep_alive<0, 1>(obj);                                             \
      return nb::handle(obj);                                                  \
    }                                                                          \
                                                                               \
    operator TYPE() { return value; }                                          \
  }

namespace nanobind::detail
{
PETSC_CASTER_MACRO(DM, DM, dm);
PETSC_CASTER_MACRO(IS, IS, is);
PETSC_CASTER_MACRO(KSP, KSP, ksp);
PETSC_CASTER_MACRO(Mat, Mat, mat);
PETSC_CASTER_MACRO(SNES, SNES, snes);
PETSC_CASTER_MACRO(Vec, Vec, vec);
} // namespace nanobind::detail

#undef PETSC_CASTER_MACRO
