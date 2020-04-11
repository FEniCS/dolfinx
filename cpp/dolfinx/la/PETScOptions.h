// Copyright (C) 2013-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <boost/lexical_cast.hpp>
#include <petscoptions.h>
#include <string>

namespace dolfinx::la
{

/// These class provides static functions that permit users to set and
/// retrieve PETSc options via the PETSc option/parameter system. The
/// option must not be prefixed by '-', e.g.
///
///     PETScOptions::set("mat_mumps_icntl_14", 40);

class PETScOptions
{
public:
  /// Set PETSc option that takes no value
  static void set(std::string option);

  /// Generic function for setting PETSc option
  template <typename T>
  static void set(std::string option, const T value)
  {
    if (option[0] != '-')
      option = '-' + option;

    PetscErrorCode ierr;
    ierr
        = PetscOptionsSetValue(nullptr, option.c_str(),
                               boost::lexical_cast<std::string>(value).c_str());
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "PetscOptionsSetValue");
  }

  /// Clear a PETSc option
  static void clear(std::string option);

  /// Clear PETSc global options database
  static void clear();
};
} // namespace dolfinx::la
