// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifndef __DOLFIN_PETSC_OPTIONS_H
#define __DOLFIN_PETSC_OPTIONS_H

#ifdef HAS_PETSC

#include <string>
#include <boost/lexical_cast.hpp>
#include <petscoptions.h>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/log/log.h>
#include "PETScObject.h"

namespace dolfin
{
  /// These class provides static functions that permit users to set
  /// and retrieve PETSc options via the PETSc option/parameter
  /// system. The option must not be prefixed by '-', e.g.
  ///
  ///     PETScOptions::set("mat_mumps_icntl_14", 40);

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
    template<typename T>
      static void set(std::string option, const T value)
    {
      SubSystemsManager::init_petsc();

      if (option[0] != '-')
        option = '-' + option;

      PetscErrorCode ierr;
      #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 6 && PETSC_VERSION_RELEASE == 1
      ierr = PetscOptionsSetValue(option.c_str(),
               boost::lexical_cast<std::string>(value).c_str());
      #else
      ierr = PetscOptionsSetValue(NULL, option.c_str(),
               boost::lexical_cast<std::string>(value).c_str());
      #endif
      if (ierr != 0)
        PETScObject::petsc_error(ierr, __FILE__, "PetscOptionsSetValue");
    }

    /// Clear PETSc option
    static void clear(std::string option);

  };
}

#endif
#endif
