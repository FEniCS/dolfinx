// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "PETScOptions.h"

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option)
{
  PETScOptions::set<std::string>(option, "");
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option, bool value)
{
  PETScOptions::set<bool>(option, value);
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option, int value)
{
  PETScOptions::set<int>(option, value);
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option, double value)
{
  PETScOptions::set<double>(option, value);
}
//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option, std::string value)
{
  PETScOptions::set<std::string>(option, value);
}
//-----------------------------------------------------------------------------
void PETScOptions::clear(std::string option)
{
  common::SubSystemsManager::init_petsc();

  if (option[0] != '-')
    option = '-' + option;

  PetscErrorCode ierr;
  ierr = PetscOptionsClearValue(NULL, option.c_str());
  if (ierr != 0)
    PETScObject::petsc_error(ierr, __FILE__, "PetscOptionsClearValue");
}
//-----------------------------------------------------------------------------
void PETScOptions::clear()
{
  common::SubSystemsManager::init_petsc();
  PetscErrorCode ierr = PetscOptionsClear(NULL);
  if (ierr != 0)
    PETScObject::petsc_error(ierr, __FILE__, "PetscOptionsClear");
}
//-----------------------------------------------------------------------------

#endif
