// Copyright (C) 2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-01-07
// Last changed:

#ifndef __PETSC_OBJECT_H
#define __PETSC_OBJECT_H

#include <dolfin/SubSystemsManager.h>

namespace dolfin
{ 
  /// This class calls SubSystemsManger to initialise PETSc.
  ///
  /// All PETSc objects must be derived from this class.

  class PETScObject
  {
  public:

    PETScObject() { SubSystemsManager::initPETSc(); }

  };

}

#endif

