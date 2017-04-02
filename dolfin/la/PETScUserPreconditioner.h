// Copyright (C) 2005 Johan Jansson
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
//
// Modified by Anders Logg 2005-2006.
// Modified Garth N. Wells 2005
//
// First added:  2005
// Last changed: 2008-01-07

#ifndef __PETSC_USER_PRECONDITIONER_H
#define __PETSC_USER_PRECONDITIONER_H

#ifdef HAS_PETSC

#include <petscksp.h>
#include <petscpc.h>
#include "PETScObject.h"

namespace dolfin
{

  class PETScVector;

  /// This class specifies the interface for user-defined Krylov
  /// method PETScPreconditioners. A user wishing to implement her own
  /// PETScPreconditioner needs only supply a function that approximately
  /// solves the linear system given a right-hand side.

  class PETScUserPreconditioner : public PETScObject
  {
  public:

    /// Constructor
    PETScUserPreconditioner();

    /// Destructor
    virtual ~PETScUserPreconditioner();

    /// Set up
    static void setup(const KSP ksp, PETScUserPreconditioner& pc);

    /// Solve linear system approximately for given right-hand side b
    virtual void solve(PETScVector& x, const PETScVector& b) = 0;

  protected:

    /// PETSc PC object
    PC petscpc;

  private:

    static int PCApply(PC pc, Vec x, Vec y);
    static int PCCreate(PC pc);

  };

}

#endif

#endif
