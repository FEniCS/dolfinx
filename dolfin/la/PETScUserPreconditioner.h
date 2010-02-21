// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
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

    static void setup(const KSP ksp, PETScUserPreconditioner& pc);

    /// Solve linear system approximately for given right-hand side b
    virtual void solve(PETScVector& x, const PETScVector& b) = 0;

  protected:

    PC petscpc;

  private:

    static int PCApply(PC pc, Vec x, Vec y);
    static int PCCreate(PC pc);

  };

}

#endif

#endif
