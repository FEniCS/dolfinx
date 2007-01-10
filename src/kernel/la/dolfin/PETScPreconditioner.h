// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005-2006.
// Modified Garth N. Wells 2005
//
// First added:  2005
// Last changed: 2006-08-15

#ifndef __PETSC_PRECONDITIONER_H
#define __PETSC_PRECONDITIONER_H

#ifdef HAVE_PETSC_H

#include <dolfin/Preconditioner.h>
#include <dolfin/PETScManager.h>

namespace dolfin
{

  class PETScVector;

  /// This class specifies the interface for user-defined Krylov
  /// method PETScPreconditioners. A user wishing to implement her own
  /// PETScPreconditioner needs only supply a function that approximately
  /// solves the linear system given a right-hand side.

  class PETScPreconditioner
  {
  public:

    /// Constructor
    PETScPreconditioner();

    /// Destructor
    virtual ~PETScPreconditioner();

    static void setup(const KSP ksp, PETScPreconditioner &pc);

    /// Solve linear system approximately for given right-hand side b
    virtual void solve(PETScVector& x, const PETScVector& b) = 0;

    /// Friends
    friend class PETScKrylovSolver;

  protected:

    PC petscpc;

  private:

    static int PCApply(PC pc, Vec x, Vec y);
    static int PCCreate(PC pc);

    /// Return PETSc PETScPreconditioner type
    static PCType getType(Preconditioner pc);

  };

}

#endif

#endif
