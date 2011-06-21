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
// Last changed: 2011-01-15

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

  class NonlinearVariationalSolver : public Variable
  {
  public:

    /// Create linear variational solver
    NonlinearVariationalSolver(const Form& F,
                               const Form& J,
                               Function& u,
                               std::vector<const BoundaryCondition*> bcs);

    /// Create linear variational solver
    NonlinearVariationalSolver(boost::shared_ptr<const Form> F,
                               boost::shared_ptr<const Form> J,
                               boost::shared_ptr<Function> u,
                               std::vector<boost::shared_ptr<const BoundaryCondition> > bcs);

    /// Solve variational problem
    void solve();

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("nonlinear_variational_solver");

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

    // Check forms
    void check_forms() const;

    // The residual form
    boost::shared_ptr<const Form> F;

    // The Jacobian form
    boost::shared_ptr<const Form> J;

    // The solution
    boost::shared_ptr<Function> u;

    // The boundary conditions
    std::vector<boost::shared_ptr<const BoundaryCondition> > bcs;

    // Nonlinear (algebraic) problem
    class _NonlinearProblem : public NonlinearProblem
    {
    public:

      // Constructor
      _NonlinearProblem(const NonlinearVariationalSolver& solver);

      // Destructor
      ~_NonlinearProblem();

      // Compute F at current point x
      virtual void F(GenericVector& b, const GenericVector& x);

      // Compute J = F' at current point x
      virtual void J(GenericMatrix& A, const GenericVector& x);

    private:

      // Reference to variational solver
      const NonlinearVariationalSolver& solver;

      // True if Jacobian has been initialized
      bool jacobian_initialized;

    };

    // The Newton problem is a friend
    friend class _NonlinearProblem;


  };

}

#endif
