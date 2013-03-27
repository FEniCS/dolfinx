// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Marie E. Rognes, 2011.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem.h)
// Last changed: 2011-10-20

#ifndef __NONLINEAR_VARIATIONAL_SOLVER_H
#define __NONLINEAR_VARIATIONAL_SOLVER_H

#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/nls/PETScSNESSolver.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include "NonlinearVariationalProblem.h"

namespace dolfin
{

  /// This class implements a solver for nonlinear variational problems.

  class NonlinearVariationalSolver : public Variable
  {
  public:

    /// Create nonlinear variational solver for given problem
    NonlinearVariationalSolver(NonlinearVariationalProblem& problem);

    /// Create nonlinear variational solver for given problem (shared pointer version)
    NonlinearVariationalSolver(boost::shared_ptr<NonlinearVariationalProblem> problem);

    /// Solve variational problem
    ///
    /// *Returns*
    ///     std::pair<std::size_t, bool>
    ///         Pair of number of Newton iterations, and whether
    ///         iteration converged)
    std::pair<std::size_t, bool> solve();

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("nonlinear_variational_solver");

      p.add("linear_solver", "default");
      p.add("preconditioner", "default");
      p.add("symmetric", false);
      p.add("reset_jacobian", true);

      std::set<std::string> nonlinear_solvers;
      nonlinear_solvers.insert("newton");
      std::string default_nonlinear_solver = "newton";

      p.add("print_rhs", false);
      p.add("print_matrix", false);

      p.add(NewtonSolver::default_parameters());
      p.add(LUSolver::default_parameters());
      p.add(KrylovSolver::default_parameters());

      #ifdef HAS_PETSC
      p.add(PETScSNESSolver::default_parameters());
      nonlinear_solvers.insert("snes");
      #endif

      p.add("nonlinear_solver", default_nonlinear_solver, nonlinear_solvers);

      return p;
    }

  private:

    // Nonlinear (algebraic) problem
    class NonlinearDiscreteProblem : public NonlinearProblem
    {
    public:

      // Constructor
      NonlinearDiscreteProblem(boost::shared_ptr<NonlinearVariationalProblem> problem,
                               boost::shared_ptr<NonlinearVariationalSolver> solver);

      // Destructor
      ~NonlinearDiscreteProblem();

      // Compute F at current point x
      virtual void F(GenericVector& b, const GenericVector& x);

      // Compute J = F' at current point x
      virtual void J(GenericMatrix& A, const GenericVector& x);

    private:

      // Problem and solver objects
      boost::shared_ptr<NonlinearVariationalProblem> _problem;
      boost::shared_ptr<NonlinearVariationalSolver> _solver;

      // True if Jacobian has been initialized
      bool jacobian_initialized;

    };

    // The nonlinear problem
    boost::shared_ptr<NonlinearVariationalProblem> _problem;

    // The nonlinear discrete problem
    boost::shared_ptr<NonlinearDiscreteProblem> nonlinear_problem;

    // The Newton solver
    boost::shared_ptr<NewtonSolver> newton_solver;

    #ifdef HAS_PETSC
    // Or, alternatively, the SNES solver
    boost::shared_ptr<PETScSNESSolver> snes_solver;
    #endif

  };

}

#endif
