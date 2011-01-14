// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Marie E. Rognes, 2011.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem.h)
// Last changed: 2011-01-14

#ifndef __NONLINEAR_VARIATIONAL_SOLVER_H
#define __NONLINEAR_VARIATIONAL_SOLVER_H

#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>

namespace dolfin
{

  class Function;
  class VariationalProblem;
  class Parameters;

  /// This class implements a solver for nonlinear variational problems.

  class NonlinearVariationalSolver
  {
  public:

    /// Solve variational problem
    static void solve(Function& u,
                      const VariationalProblem& problem,
                      const Parameters& parameters);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("variational_problem");

      p.add("linear_solver",  "lu");
      p.add("preconditioner", "default");
      p.add("symmetric", false);
      p.add("reset_jacobian", true);

      p.add("print_rhs", false);
      p.add("print_matrix", false);

      p.add(NewtonSolver::default_parameters());
      p.add(LUSolver::default_parameters());
      p.add(KrylovSolver::default_parameters());

      return p;
    }

  private:

    // Nonlinear (algebraic) problem
    class _NonlinearProblem : public NonlinearProblem
    {
    public:

      // Constructor
      _NonlinearProblem(const VariationalProblem& problem,
                        const Parameters& parameters);

      // Destructor
      ~_NonlinearProblem();

      // Compute F at current point x
      virtual void F(GenericVector& b, const GenericVector& x);

      // Compute J = F' at current point x
      virtual void J(GenericMatrix& A, const GenericVector& x);

    private:

      // Reference to variational problem being solved
      const VariationalProblem& problem;

      // Reference to solver parameters
      const Parameters& parameters;

      // True if Jacobian has been initialized
      bool jacobian_initialized;

    };

  };

}

#endif
