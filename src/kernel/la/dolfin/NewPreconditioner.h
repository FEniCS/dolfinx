// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_PRECONDITIONER_H
#define __NEW_PRECONDITIONER_H

#include <petsc/petscpc.h>
#include <dolfin/pcimpl.h>

namespace dolfin
{

  class NewVector;

  /// This class specifies the interface for user-defined Krylov
  /// method preconditioners. A user wishing to implement her own
  /// preconditioner needs only supply a function that approximately
  /// solves the linear system given a right-hand side.

  class NewPreconditioner
  {
  public:

    /// Constructor
    NewPreconditioner();

    /// Destructor
    virtual ~NewPreconditioner();

    /// Solve linear system approximately for given right-hand side b
    virtual void solve(NewVector& x, const NewVector& b) = 0;

    static int PCApply(PC pc, Vec x, Vec y);

  protected:

    PC petscpc;

  };

}

#endif
