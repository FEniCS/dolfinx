// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PRECONDITIONER_H
#define __PRECONDITIONER_H

#include <petscpc.h>

namespace dolfin
{

  class Vector;

  /// This class specifies the interface for user-defined Krylov
  /// method preconditioners. A user wishing to implement her own
  /// preconditioner needs only supply a function that approximately
  /// solves the linear system given a right-hand side.

  class Preconditioner
  {
  public:

    /// Constructor
    Preconditioner();

    /// Destructor
    virtual ~Preconditioner();

    /// Solve linear system approximately for given right-hand side b
    virtual void solve(Vector& x, const Vector& b) = 0;

    static int PCApply(PC pc, Vec x, Vec y);
    static int PCCreate(PC pc);

  protected:

    PC petscpc;

  };

}

#endif
