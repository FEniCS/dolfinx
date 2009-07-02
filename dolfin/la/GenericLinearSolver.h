// Copyright (C) 2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2009.
//
// First added:  2008-08-26
// Last changed: 2009-06-29

#ifndef __GENERIC_LINEAR_SOLVER_H
#define __GENERIC_LINEAR_SOLVER_H

#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;

  /// This class provides a general solver for linear systems Ax = b.

  class GenericLinearSolver : public Variable
  {
  public:

    /// Solve linear system Ax = b
    virtual uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b) = 0;

  };

}

#endif
