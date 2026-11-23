// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <petscsys.h>

namespace dolfinx_test
{
/// Install once after PETSc initialisation to suppress expected tracebacks.
inline void set_petsc_error_handler()
{
  static const PetscErrorCode error
      = PetscPushErrorHandler(PetscReturnErrorHandler, nullptr);
  REQUIRE(error == PETSC_SUCCESS);
}
} // namespace dolfinx_test
