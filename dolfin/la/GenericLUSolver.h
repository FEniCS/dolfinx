// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-07-11
// Last changed:

#ifndef __GENERIC_LU_SOLVER_H
#define __GENERIC_LU_SOLVER_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  class GenericMatrix;

  /// This a base class for LU solvers

  class GenericLUSolver : public GenericLinearSolver
  {

  public:

    /// Set operator (matrix)
    virtual void set_operator(const GenericMatrix& A) = 0;

    /// Solve linear system Ax = b
    virtual uint solve(GenericVector& x, const GenericVector& b) = 0;

    /// Factor the sparse matrix A
    virtual void factorize() = 0;

    /// Solve factorized system
    virtual uint solve_factorized(GenericVector& x, const GenericVector& b) const = 0;

    /// Solve linear system Ax = b
    virtual uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    { error("solve(A, x, b) is not implemented. Consider trying solve(x, b)."); return 0; }

  };

}

#endif
