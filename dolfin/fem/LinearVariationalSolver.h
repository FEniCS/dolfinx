// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Marie E. Rognes, 2011.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem)
// Last changed: 2011-01-14

#ifndef __LINEAR_VARIATIONAL_SOLVER_H
#define __LINEAR_VARIATIONAL_SOLVER_H

#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>

namespace dolfin
{

  class Function;
  class VariationalProblem;
  class Parameters;

  /// This class implements a solver for linear variational problems.

  class LinearVariationalSolver
  {
  public:

    /// Solve variational problem
    static void solve(Function& u,
                      const VariationalProblem& problem,
                      const Parameters& parameters);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("linear_variational_solver");

      p.add("linear_solver", "lu");
      p.add("preconditioner", "default");
      p.add("symmetric", false);
      p.add("reset_jacobian", true);

      p.add("print_rhs", false);
      p.add("print_matrix", false);

      p.add(LUSolver::default_parameters());
      p.add(KrylovSolver::default_parameters());

      return p;
    }

  };

}

#endif
