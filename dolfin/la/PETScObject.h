// Copyright (C) 2008 Garth N. Wells
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

#ifndef __PETSC_OBJECT_H
#define __PETSC_OBJECT_H

#include <string>
#include <dolfin/common/SubSystemsManager.h>

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
    static void petsc_error(int error_code,
                            std::string filename,
                            std::string petsc_function);
  };
}

#endif
