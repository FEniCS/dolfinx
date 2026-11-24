// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <exception>
#include <petscsys.h>
#include <utility>
#ifdef HAS_SLEPC
#include <slepcsys.h>
#endif

namespace dolfinx_test
{
/// Run a callable that is expected to provoke a PETSc error without
/// printing its traceback.
template <typename F>
void without_petsc_traceback(F&& f)
{
  REQUIRE(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr)
          == PETSC_SUCCESS);
  std::exception_ptr exception;
  try
  {
    std::forward<F>(f)();
  }
  catch (...)
  {
    exception = std::current_exception();
  }
  REQUIRE(PetscPopErrorHandler() == PETSC_SUCCESS);
  if (exception)
    std::rethrow_exception(exception);
}

/// Initialise PETSc (a no-op if already initialised).
inline void init_petsc()
{
  int argc = 0;
  char** argv = nullptr;
  REQUIRE(PetscInitialize(&argc, &argv, nullptr, nullptr) == PETSC_SUCCESS);
}

#ifdef HAS_SLEPC
/// Initialise SLEPc (a no-op if already initialised).
inline void init_slepc()
{
  int argc = 0;
  char** argv = nullptr;
  REQUIRE(SlepcInitialize(&argc, &argv, nullptr, nullptr) == PETSC_SUCCESS);
}
#endif
} // namespace dolfinx_test
