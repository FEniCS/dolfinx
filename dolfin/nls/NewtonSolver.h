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
// Modified by Anders E. Johansen 2011.
//
// First added:  2005-10-23
// Last changed: 2011-07-11

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <utility>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class GenericLinearSolver;
  class GenericLinearAlgebraFactory;
  class GenericMatrix;
  class GenericVector;
  class NonlinearProblem;

  /// This class defines a Newton solver for nonlinear systems of
  /// equations of the form :math:`F(x) = 0`.

  class NewtonSolver : public Variable
  {
  public:

    /// Create nonlinear solver with default linear solver and default
    /// linear algebra backend
    NewtonSolver(std::string solver_type="lu",
                 std::string pc_type="default");

    /// Create nonlinear solver using provided linear solver and linear algebra
    /// backend determined by factory
    ///
    /// *Arguments*
    ///     solver (_GenericLinearSolver_)
    ///         The linear solver.
    ///     factory (_GenericLinearAlgebraFactory_)
    ///         The factory.
    NewtonSolver(boost::shared_ptr<GenericLinearSolver> solver,
                 GenericLinearAlgebraFactory& factory);

    /// Destructor
    virtual ~NewtonSolver();

    /// Solve abstract nonlinear problem :math:`F(x) = 0` for given
    /// :math:`F` and Jacobian :math:`\dfrac{\partial F}{\partial x}`.
    ///
    /// *Arguments*
    ///     nonlinear_function (_NonlinearProblem_)
    ///         The nonlinear problem.
    ///     x (_GenericVector_)
    ///         The vector.
    ///
    /// *Returns*
    ///     std::pair<uint, bool>
    ///         Pair of number of Newton iterations, and whether
    ///         iteration converged)
    std::pair<uint, bool> solve(NonlinearProblem& nonlinear_function,
                                GenericVector& x);

    /// Return Newton iteration number
    ///
    /// *Returns*
    ///     uint
    ///         The iteration number.
    uint iteration() const;

    /// Return current residual
    ///
    /// *Returns*
    ///     double
    ///         Current residual.
    double residual() const;

    /// Return current relative residual
    ///
    /// *Returns*
    ///     double
    ///       Current relative residual.
    double relative_residual() const;

    /// Return the linear solver
    ///
    /// *Returns*
    ///     _GenericLinearSolver_
    ///         The linear solver.
    GenericLinearSolver& linear_solver() const;

    /// Default parameter values
    ///
    /// *Returns*
    ///     _Parameters_
    ///         Parameter values.
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
    boost::shared_ptr<GenericVector> dx;

    /// Resdiual vector
    boost::shared_ptr<GenericVector> b;

  };

}

#endif
