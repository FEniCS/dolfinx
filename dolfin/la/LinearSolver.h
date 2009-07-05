// Copyright (C) 2004-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
// Modified by Ola Skavhaug 2008.
//
// First added:  2004-06-19
// Last changed: 2009-06-30

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

#include <dolfin/common/types.h>
#include "LUSolver.h"
#include "KrylovSolver.h"
#include "GenericLinearSolver.h"

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;

  /// This class provides a general solver for linear systems Ax = b.

  class LinearSolver : public GenericLinearSolver
  {
  public:

    /// Create linear solver
    LinearSolver(std::string solver_type = "lu", std::string pc_type = "ilu");

    /// Destructor
    ~LinearSolver();

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("ode");

      p.add(LUSolver::default_parameters());
      p.add(KrylovSolver::default_parameters());

      return p;
    }

  private:

    // LU solver
    LUSolver* lu_solver;

    // Krylov solver
    KrylovSolver* krylov_solver;

  };

}

#endif

