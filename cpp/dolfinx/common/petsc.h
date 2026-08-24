// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include <format>
#include <petscsys.h>
#include <source_location>
#include <string>
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

/// @brief Set a PETSc option in the PETSc options/parameter database.
/// The option must not be prefixed by '-', e.g.
///
///     common::petsc::set_option("mat_mumps_icntl_14", 40);
void set_option(std::string option);

/// @brief Set a PETSc option that takes a value in the PETSc
/// options/parameter database. The option must not be prefixed by
/// '-', e.g.
///
///     common::petsc::set_option("mat_mumps_icntl_14", 40);
template <typename T>
  requires requires(const T& value) { std::format("{}", value); }
void set_option(std::string option, const T& value)
{
  if (option[0] != '-')
    option = '-' + option;

  check(PetscOptionsSetValue(nullptr, option.c_str(),
                             std::format("{}", value).c_str()),
        "PetscOptionsSetValue");
}

/// @brief Clear a PETSc option from the PETSc options/parameter
/// database.
void clear_option(std::string option);

/// @brief Clear the PETSc global options/parameter database.
void clear_options();
} // namespace petsc
} // namespace dolfinx::common

#endif
