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
//
// First added:  2013-08-12
// Last changed:

#ifndef __DOLFIN_PETSC_OPTIONS_H
#define __DOLFIN_PETSC_OPTIONS_H

#ifdef HAS_PETSC

#include <string>
#include <boost/lexical_cast.hpp>
#include <petscoptions.h>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  /// These function permit users set and retreive PETSc options via
  /// the PETSc parameter systems

  void set_petsc_option(std::string option);
  void set_petsc_option(std::string option, bool value);
  void set_petsc_option(std::string option, int value);
  void set_petsc_option(std::string option, double value);
  void set_petsc_option(std::string option, std::string value);

  template<typename T>
    void set_petsc_option(std::string option, const T value)
  {
    SubSystemsManager::init_petsc();

    PetscErrorCode ierr;
    std::string _option = "-" + option;
    ierr = PetscOptionsSetValue(_option.c_str(),
                            boost::lexical_cast<std::string>(value).c_str());
    if (ierr != 0)
    {
      dolfin_error("petsc_settings.h",
                  "set PETSc option/parameter '" + option + "'",
                  "PETSc error code is: %d", ierr);

    }
  }

}

#endif
#endif
