// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScOptions.h"

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
void PETScOptions::set(std::string option)
{
  PETScOptions::set<std::string>(option, "");
}
//-----------------------------------------------------------------------------
void PETScOptions::clear(std::string option)
{
  if (option[0] != '-')
    option = '-' + option;

  PetscErrorCode ierr;
  ierr = PetscOptionsClearValue(NULL, option.c_str());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "PetscOptionsClearValue");
}
//-----------------------------------------------------------------------------
void PETScOptions::clear()
{
  PetscErrorCode ierr = PetscOptionsClear(NULL);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "PetscOptionsClear");
}
//-----------------------------------------------------------------------------
