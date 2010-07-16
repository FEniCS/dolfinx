// Copyright (C) 2008-2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2009.
//
// First added:  2008-08-26
// Last changed: 2009-06-29
// Last changed: 2010-07-16

#ifndef __GENERIC_LINEAR_SOLVER_H
#define __GENERIC_LINEAR_SOLVER_H

#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;

  /// This class provides a general solver for linear systems Ax = b.

  class GenericLinearSolver : public Variable
  {
  public:

    /// Solve the operator (matrix)
    virtual void set_operator(const GenericMatrix& A)
    { error("set_operator(A) is not implemented."); }

    /// Solve linear system Ax = b
    virtual uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    { error("solve(A, x, b) is not implemented. Consider trying solve(x, b)."); return 0; }

    /// Solve linear system Ax = b
    virtual uint solve(GenericVector& x, const GenericVector& b)
    { error("solve(x, b) is not yet implemented for this backend."); return 0; }

  };

}

#endif
