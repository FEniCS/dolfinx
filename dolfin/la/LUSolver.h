// Copyright (C) 2007-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Dag Lindbo, 2008.
// Modified by Anders Logg, 2008-2009.
// Modified by Kent-Andre Mardal, 2008.
//
// First added:  2007-07-03
// Last changed: 2009-07-07

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <boost/scoped_ptr.hpp>
#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "CholmodCholeskySolver.h"
#include "UmfpackLUSolver.h"
#include "PETScLUSolver.h"
#include "EpetraLUSolver.h"
#include "GenericLinearSolver.h"

namespace dolfin
{

  class LUSolver : public GenericLUSolver
  {

  /// LU solver for the built-in LA backends. The type can be "lu" or
  /// "cholesky". Cholesky is suitable only for symmetric positive-definite
  /// matrices. Cholesky is not yet suppprted for all backends (which will
  /// default to LU.

  public:

    LUSolver(std::string type = "lu")
    {
      // Set default parameters
      parameters = default_parameters();

      // Create suitable solver
      const std::string backend = parameters["linear_algebra_backend"];
      if (backend == "uBLAS" || backend == "MTL4")
        solver.reset(new UmfpackLUSolver());
      else if (backend == "PETSc")
        solver.reset(new PETScLUSolver());
      else if (backend == "Epetra")
        error("EpetraLUSolver needs to be updated..");
        //solver.reset(new EpetraLUSolver());
      else
        error("No suitable LU solver for linear algebra backend.");

      solver->parameters.update(parameters);
    }

    ~LUSolver()
    {
      // Do nothing
    }

    /// Set operator (matrix)
    void set_operator(const GenericMatrix& A)
    {
      assert(solver);
      solver->set_operator(A);
    }

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b)
    {
      assert(solver);
      return solver->solve(x, b);
    }

    /// Factor the sparse matrix A
    void factorize()
    {
      assert(solver);
      solver->factorize();
    }

    uint solve_factorized(GenericVector& x, const GenericVector& b) const
    {
      assert(solver);
      return solver->solve_factorized(x, b);
    }

    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    {
      Timer timer("LU solver");
      return solver->solve(A, x, b);
    }

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("lu_solver");
      p.add("report", true);
      p.add("same_nonzero_pattern", false);
      return p;
    }

  private:

    // Solver
    boost::scoped_ptr<GenericLUSolver> solver;

  };
}

#endif
