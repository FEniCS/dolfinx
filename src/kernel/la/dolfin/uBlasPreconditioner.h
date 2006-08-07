// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-06-23
// Last changed: 2006-07-04

#ifndef __UBLAS_PRECONDITIONER_H
#define __UBLAS_PRECONDITIONER_H

#include <dolfin/Parametrized.h>

namespace dolfin
{

  class uBlasVector;

  /// This class specifies the interface for preconditioners for the
  /// uBlas Krylov solver.

  class uBlasPreconditioner : public Parametrized
  {
  public:

    /// Constructor
    uBlasPreconditioner() {};

    /// Destructor
    virtual ~uBlasPreconditioner() {};

    /// Solve linear system (M^-1)Ax = y
    virtual void solve(uBlasVector& x, const uBlasVector& b) const = 0;

  };

}

#endif
