// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScOptions.h"
#include "PETScVector.h"

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
void petsc::Options::set(std::string option)
{
  petsc::Options::set<std::string>(option, "");
}
//-----------------------------------------------------------------------------
void petsc::Options::clear(std::string option)
{
  if (option[0] != '-')
    option = '-' + option;

  PetscErrorCode ierr;
  ierr = PetscOptionsClearValue(nullptr, option.c_str());
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsClearValue");
}
//-----------------------------------------------------------------------------
void petsc::Options::clear()
{
  PetscErrorCode ierr = PetscOptionsClear(nullptr);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsClear");
}
//-----------------------------------------------------------------------------
