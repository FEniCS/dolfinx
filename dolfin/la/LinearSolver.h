// Copyright (C) 2004-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006-2010.
// Modified by Ola Skavhaug 2008.
//
// First added:  2004-06-19
// Last changed: 2010-07-16

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

#include <string>
#include <boost/scoped_ptr.hpp>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;
  class LUSolver;
  class KrylovSolver;

  /// This class provides a general solver for linear systems Ax = b.

  class LinearSolver : public GenericLinearSolver
  {
  public:

    /// Create linear solver
    LinearSolver(std::string solver_type = "lu", std::string pc_type = "ilu");

    /// Destructor
    ~LinearSolver();

    /// Set the operator (matrix)
    void set_operator(const GenericMatrix& A);

    /// Set the operator (matrix) and preconitioner matrix
    void set_operators(const GenericMatrix& A, const GenericMatrix& P);

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("linear_solver");
      return p;
    }

  private:

    // Solver
    boost::scoped_ptr<GenericLinearSolver> solver;

  };

}

#endif

