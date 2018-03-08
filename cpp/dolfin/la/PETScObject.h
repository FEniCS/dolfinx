// Copyright (C) 2008 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include <string>

namespace dolfin
{
namespace la
{

/// All PETSc objects must be derived from this class.

class PETScObject
{
public:

  /// Destructor
  virtual ~PETScObject() {}

  /// Print error message for PETSc calls that return an error
  static void petsc_error(int error_code, std::string filename,
                          std::string petsc_function);
};
}
}
#endif
