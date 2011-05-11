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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-01-07
// Last changed: 2011-01-24

#ifndef __PETSC_OBJECT_H
#define __PETSC_OBJECT_H

#include <dolfin/common/SubSystemsManager.h>

namespace dolfin
{
  /// This class calls SubSystemsManager to initialise PETSc.
  ///
  /// All PETSc objects must be derived from this class.

  class PETScObject
  {
  public:

    PETScObject() { SubSystemsManager::init_petsc(); }

  };

}

#endif
