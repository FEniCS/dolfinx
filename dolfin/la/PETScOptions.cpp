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

#ifdef HAS_PETSC

#include "PETScOptions.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScOptions::PETScOptions() : PETScObject()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScOptions::~PETScOptions()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option)
{
  set<std::string>(option, "");
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option, bool value)
{
  set<bool>(option, value);
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option, int value)
{
  set<int>(option, value);
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option, double value)
{
  set<double>(option, value);
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option, std::string value)
{
  set<std::string>(option, value);
}
//-----------------------------------------------------------------------------
void PETScOptions::clear(std::string option)
{
  if (option[0] != '-')
    option = '-' + option;

  PetscErrorCode ierr;
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 6 && PETSC_VERSION_RELEASE == 1
  ierr = PetscOptionsClearValue(option.c_str());
  #else
  ierr = PetscOptionsClearValue(NULL, option.c_str());
  #endif
  if (ierr != 0) petsc_error(ierr, __FILE__, "PetscOptionsClearValue");
}
//-----------------------------------------------------------------------------

#endif
