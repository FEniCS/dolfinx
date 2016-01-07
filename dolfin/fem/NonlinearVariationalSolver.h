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
// Modified by Corrado Maurini, 2013.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem.h)
// Last changed: 2013-11-20

#ifndef __NONLINEAR_VARIATIONAL_SOLVER_H
#define __NONLINEAR_VARIATIONAL_SOLVER_H

#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/nls/PETScSNESSolver.h>
#include "NonlinearVariationalProblem.h"
#include "SystemAssembler.h"

namespace dolfin
{

  /// This class implements a solver for nonlinear variational
  /// problems.

  class NonlinearVariationalSolver : public Variable
  {
  public:

    /// Create nonlinear variational solver for given problem
    explicit NonlinearVariationalSolver(std::shared_ptr<NonlinearVariationalProblem> problem);

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

      p.add("symmetric", false);
      p.add("print_rhs", false);
      p.add("print_matrix", false);

      std::set<std::string> nonlinear_solvers = {"newton"};
      std::string default_nonlinear_solver = "newton";
      p.add(NewtonSolver::default_parameters());

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
      NonlinearDiscreteProblem(
        std::shared_ptr<const NonlinearVariationalProblem> problem,
        std::shared_ptr<const NonlinearVariationalSolver> solver);

      // Destructor
      ~NonlinearDiscreteProblem();

      // Compute F at current point x
      virtual void F(GenericVector& b, const GenericVector& x);

      // Compute J = F' at current point x
      virtual void J(GenericMatrix& A, const GenericVector& x);

    private:

      // Problem and solver objects
      std::shared_ptr<const NonlinearVariationalProblem> _problem;
      std::shared_ptr<const NonlinearVariationalSolver> _solver;

    };

    // The nonlinear problem
    std::shared_ptr<NonlinearVariationalProblem> _problem;

    // The nonlinear discrete problem
    std::shared_ptr<NonlinearDiscreteProblem> nonlinear_problem;

    // The Newton solver
    std::shared_ptr<NewtonSolver> newton_solver;

    #ifdef HAS_PETSC
    // Or, alternatively, the SNES solver
    std::shared_ptr<PETScSNESSolver> snes_solver;
    #endif

  };

}

#endif
