// Copyright (C) 2005-2008 Garth N. Wells
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
// Modified by Anders Logg 2006-2011.
//
// First added:  2005-10-23
// Last changed: 2011-03-29

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <utility>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class GenericLinearSolver;
  class LinearAlgebraFactory;
  class GenericMatrix;
  class GenericVector;
  class NonlinearProblem;

  /// This class defines a Newton solver for equations of the form F(u) = 0.

  class NewtonSolver : public Variable
  {
  public:

    /// Create nonlinear solver with default linear solver and default
    /// linear algebra backend
    NewtonSolver(std::string solver_type = "lu",
                 std::string pc_type = "default");

    /// Create nonlinear solver using provided linear solver and linear algebra
    /// backend determined by factory
    NewtonSolver(GenericLinearSolver& solver, LinearAlgebraFactory& factory);

    /// Destructor
    virtual ~NewtonSolver();

    /// Solve abstract nonlinear problem F(x) = 0 for given vector F and
    /// Jacobian dF/dx
    std::pair<uint, bool> solve(NonlinearProblem& nonlinear_function,
                                GenericVector& x);

    /// Return Newton iteration number
    uint iteration() const;

    /// Return current residual
    double residual() const;

    /// Return current relative residual
    double relative_residual() const;

    /// Return the linear solver
    GenericLinearSolver& linear_solver() const;

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Convergence test
    virtual bool converged(const GenericVector& b, const GenericVector& dx,
                           const NonlinearProblem& nonlinear_problem);

    /// Current number of Newton iterations
    uint newton_iteration;

    /// Most recent residual and intitial residual
    double _residual, residual0;

    /// Solver
    boost::shared_ptr<GenericLinearSolver> solver;

    /// Jacobian matrix
    boost::shared_ptr<GenericMatrix> A;

    /// Solution vector
    boost::scoped_ptr<GenericVector> dx;

    /// Resdiual vector
    boost::scoped_ptr<GenericVector> b;

  };

}

#endif
