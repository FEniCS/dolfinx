// Copyright (C) 2008 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include <dolfin/common/SubSystemsManager.h>
#include <string>

namespace dolfin
{

/// This class calls SubSystemsManager to initialise PETSc.
///
/// All PETSc objects must be derived from this class.

class PETScObject
{
public:
  /// Constructor. Ensures that PETSc has been initialised.
  PETScObject() { SubSystemsManager::init_petsc(); }

  /// Destructor
  virtual ~PETScObject() {}

  /// Print error message for PETSc calls that return an error
  static void petsc_error(int error_code, std::string filename,
                          std::string petsc_function);
};
}

#endif
