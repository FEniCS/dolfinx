// Copyright (C) 2004-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
// Modified by Ola Skavhaug 2008.
//
// First added:  2004-06-19
// Last changed: 2008-04-11

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

#include <dolfin/common/types.h>
#include "Matrix.h"
#include "Vector.h"

namespace dolfin
{

  /// This class defines the interfaces for default linear solvers for
  /// systems of the form Ax = b.

  class LinearSolver
  {
  public:

    /// Constructor
    LinearSolver() {}

    /// Destructor
    virtual ~LinearSolver() {}

    /// Solve linear system Ax = b
    virtual uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b) = 0;

  };

}

#endif
