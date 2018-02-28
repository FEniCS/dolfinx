// Copyright (C) 2013-2016 Garth N. Wells, Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "PETScObject.h"
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/log/log.h>
#include <petsc.h>

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
void PETScObject::petsc_error(int error_code, std::string filename,
                              std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Fetch and clear PETSc error message
  const std::string msg = common::SubSystemsManager::singleton().petsc_err_msg;
  common::SubSystemsManager::singleton().petsc_err_msg = "";

  // Log detailed error info
  log::log(TRACE, "PETSc error in '%s', '%s'", filename.c_str(),
      petsc_function.c_str());
  log::log(TRACE, "PETSc error code '%d' (%s), message follows:", error_code, desc);
  // NOTE: don't put msg as variadic argument; it might get trimmed
  log::log(TRACE, std::string(78, '-'));
  log::log(TRACE, msg);
  log::log(TRACE, std::string(78, '-'));

  // Raise exception with standard error message
  log::dolfin_error(filename,
               "successfully call PETSc function '" + petsc_function + "'",
               "PETSc error code is: %d (%s)", error_code, desc);
}
//-----------------------------------------------------------------------------

#endif
