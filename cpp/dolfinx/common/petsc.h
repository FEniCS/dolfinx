// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include <petscsys.h>
#include <source_location>
#include <string_view>

namespace dolfinx::common
{
/// @brief PETSc error handling helpers shared across DOLFINx's PETSc
/// wrappers
namespace petsc
{
/// @brief Print error message for a PETSc call that returned an error
/// and throw a std::runtime_error.
/// @param[in] error_code PETSc error code
/// @param[in] petsc_function Name of the PETSc function that returned
/// `error_code`
/// @param[in] loc Call site of the failed PETSc call (captured
/// automatically; do not pass explicitly)
void error(PetscErrorCode error_code, std::string_view petsc_function,
           std::source_location loc = std::source_location::current());

/// @brief Throw a std::runtime_error via error() if `ierr` indicates a
/// PETSc call failed.
/// @param[in] ierr PETSc error code returned by `petsc_function`
/// @param[in] petsc_function Name of the PETSc function that returned
/// `ierr`
/// @param[in] loc Call site of the failed PETSc call (captured
/// automatically; do not pass explicitly)
inline void check(PetscErrorCode ierr, std::string_view petsc_function,
                  std::source_location loc = std::source_location::current())
{
  if (ierr != 0)
    error(ierr, petsc_function, loc);
}
} // namespace petsc
} // namespace dolfinx::common

#endif
