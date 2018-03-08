// Copyright (C) 2013-2018 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/log/log.h>
#include <petsc.h>

//-----------------------------------------------------------------------------
void dolfin::la::petsc_error(int error_code, std::string filename,
                             std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Fetch and clear PETSc error message
  const std::string msg = common::SubSystemsManager::singleton().petsc_err_msg;
  dolfin::common::SubSystemsManager::singleton().petsc_err_msg = "";

  // Log detailed error info
  dolfin::log::log(TRACE, "PETSc error in '%s', '%s'", filename.c_str(),
                   petsc_function.c_str());
  dolfin::log::log(
      TRACE, "PETSc error code '%d' (%s), message follows:", error_code, desc);
  // NOTE: don't put msg as variadic argument; it might get trimmed
  dolfin::log::log(TRACE, std::string(78, '-'));
  dolfin::log::log(TRACE, msg);
  dolfin::log::log(TRACE, std::string(78, '-'));

  // Raise exception with standard error message
  dolfin::log::dolfin_error(
      filename, "successfully call PETSc function '" + petsc_function + "'",
      "PETSc error code is: %d (%s)", error_code, desc);
}
//-----------------------------------------------------------------------------
