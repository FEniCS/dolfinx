// Copyright (C) 2017-2023 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)

#include <nanobind/nanobind.h>
#include <petsc4py/petsc4py.h>
#include <petscmat.h>
#include <petscvec.h>

// nanobind casters for PETSc/petsc4py objects

namespace nb = nanobind;

// Import petsc4py on demand
#define VERIFY_PETSC4PY_FROMPY(func)                                           \
  if (!func)                                                                   \
  {                                                                            \
    if (import_petsc4py() != 0)                                                \
      return false;                                                            \
  }

#define VERIFY_PETSC4PY_FROMCPP(func)                                          \
  if (!func)                                                                   \
  {                                                                            \
    if (import_petsc4py() != 0)                                                \
      return {};                                                               \
  }

// Macro for casting between PETSc and petsc4py objects
#define PETSC_CASTER_MACRO(TYPE, P4PYTYPE, NAME)                               \
  template <>                                                                  \
  class type_caster<_p_##TYPE>                                                 \
  {                                                                            \
  public:                                                                      \
    NB_TYPE_CASTER(TYPE, const_name(#NAME))                                    \
    bool from_python(handle src, uint8_t, cleanup_list*) noexcept              \
    {                                                                          \
      VERIFY_PETSC4PY_FROMPY(PyPetsc##P4PYTYPE##_Get);                         \
      if (PyObject_TypeCheck(src.ptr(), &PyPetsc##P4PYTYPE##_Type) != 0)       \
      {                                                                        \
        value = PyPetsc##P4PYTYPE##_Get(src.ptr());                            \
        return true;                                                           \
      }                                                                        \
      else                                                                     \
        return false;                                                          \
    }                                                                          \
                                                                               \
    static handle from_cpp(TYPE src, rv_policy policy,                         \
                           cleanup_list* /*cleanup*/) noexcept                 \
    {                                                                          \
      VERIFY_PETSC4PY_FROMCPP(PyPetsc##P4PYTYPE##_New);                        \
      if (policy == rv_policy::take_ownership)                                 \
      {                                                                        \
        PyObject* obj = PyPetsc##P4PYTYPE##_New(src);                          \
        PetscObjectDereference((PetscObject)src);                              \
        return nb::handle(obj);                                                \
      }                                                                        \
      else if (policy == rv_policy::automatic_reference                        \
               or policy == rv_policy::reference)                              \
      {                                                                        \
        PyObject* obj = PyPetsc##P4PYTYPE##_New(src);                          \
        return nb::handle(obj);                                                \
      }                                                                        \
      else                                                                     \
      {                                                                        \
        return {};                                                             \
      }                                                                        \
    }                                                                          \
                                                                               \
    operator TYPE() { return value; }                                          \
  }

namespace nanobind::detail
{
PETSC_CASTER_MACRO(Mat, Mat, mat);
PETSC_CASTER_MACRO(Vec, Vec, vec);
} // namespace nanobind::detail
#endif
