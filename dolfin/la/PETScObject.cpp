// Copyright (C) 2013-2016 Garth N. Wells, Jan Blechta
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

#include <petsc.h>

#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/log/log.h>
#include "PETScObject.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void PETScObject::petsc_error(int error_code,
                              std::string filename,
                              std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Fetch and clear PETSc error message
  const std::string msg = SubSystemsManager::singleton().petsc_err_msg;
  SubSystemsManager::singleton().petsc_err_msg = "";

  // Log detailed error info
  log(TRACE, "PETSc error in '%s', '%s'",
      filename.c_str(), petsc_function.c_str());
  log(TRACE, "PETSc error code '%d' (%s), message follows:", error_code, desc);
  // NOTE: don't put msg as variadic argument; it might get trimmed
  log(TRACE, std::string(78, '-'));
  log(TRACE, msg);
  log(TRACE, std::string(78, '-'));

  // Raise exception with standard error message
  dolfin_error(filename,
               "successfully call PETSc function '" + petsc_function + "'",
                "PETSc error code is: %d (%s)", error_code, desc);
}
//-----------------------------------------------------------------------------

#endif
