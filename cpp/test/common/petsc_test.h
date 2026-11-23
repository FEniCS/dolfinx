// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <petscsys.h>
#ifdef HAS_SLEPC
#include <slepcsys.h>
#endif

namespace dolfinx_test
{
/// Install, once, an error handler that suppresses the tracebacks of
/// PETSc errors provoked by the tests.
inline void set_petsc_error_handler()
{
  static const PetscErrorCode error
      = PetscPushErrorHandler(PetscReturnErrorHandler, nullptr);
  REQUIRE(error == PETSC_SUCCESS);
}

/// Initialise PETSc (a no-op if already initialised) and set the test
/// error handler.
inline void init_petsc()
{
  int argc = 0;
  char** argv = nullptr;
  REQUIRE(PetscInitialize(&argc, &argv, nullptr, nullptr) == PETSC_SUCCESS);
  set_petsc_error_handler();
}

#ifdef HAS_SLEPC
/// Initialise SLEPc (a no-op if already initialised) and set the test
/// error handler.
inline void init_slepc()
{
  int argc = 0;
  char** argv = nullptr;
  REQUIRE(SlepcInitialize(&argc, &argv, nullptr, nullptr) == PETSC_SUCCESS);
  set_petsc_error_handler();
}
#endif
} // namespace dolfinx_test
