// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include "PETScObject.h"
#include <boost/lexical_cast.hpp>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/log/log.h>
#include <petscoptions.h>
#include <string>

namespace dolfin
{
/// These class provides static functions that permit users to set
/// and retrieve PETSc options via the PETSc option/parameter
/// system. The option must not be prefixed by '-', e.g.
///
///     PETScOptions::set("mat_mumps_icntl_14", 40);
///
/// Note: the non-templated functions are to simplify SWIG wapping
/// into Python.

class PETScOptions
{
public:
  /// Set PETSc option that takes no value
  static void set(std::string option);

  /// Set PETSc boolean option
  static void set(std::string option, bool value);

  /// Set PETSc integer option
  static void set(std::string option, int value);

  /// Set PETSc double option
  static void set(std::string option, double value);

  /// Set PETSc string option
  static void set(std::string option, std::string value);

  /// Genetic function for setting PETSc option
  template <typename T>
  static void set(std::string option, const T value)
  {
    SubSystemsManager::init_petsc();

    if (option[0] != '-')
      option = '-' + option;

    PetscErrorCode ierr;
    ierr = PetscOptionsSetValue(
        NULL, option.c_str(), boost::lexical_cast<std::string>(value).c_str());
    if (ierr != 0)
      PETScObject::petsc_error(ierr, __FILE__, "PetscOptionsSetValue");
  }

  /// Clear a PETSc option
  static void clear(std::string option);

  /// Clear PETSc global options database
  static void clear();
};
}

#endif

