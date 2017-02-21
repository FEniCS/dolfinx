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
// Modified by Anders Logg 2006-2011
// Modified by Anders E. Johansen 2011
//
// First added:  2005-10-23
// Last changed: 2013-11-20

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <utility>
#include <memory>
#include <dolfin/common/MPI.h>
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

    /// Create nonlinear solver
    explicit NewtonSolver(MPI_Comm comm=MPI_COMM_WORLD);

    /// Create nonlinear solver using provided linear solver
    ///
    /// *Arguments*
    ///     comm (_MPI_Ccmm_)
    ///         The MPI communicator.
    ///     solver (_GenericLinearSolver_)
    ///         The linear solver.
    ///     factory (_GenericLinearAlgebraFactory_)
    ///         The factory.
    NewtonSolver(MPI_Comm comm, std::shared_ptr<GenericLinearSolver> solver,
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
    ///     std::pair<std::size_t, bool>
    ///         Pair of number of Newton iterations, and whether
    ///         iteration converged)
    std::pair<std::size_t, bool> solve(NonlinearProblem& nonlinear_function,
                                       GenericVector& x);

    /// Return Newton iteration number
    ///
    /// *Returns*
    ///     std::size_t
    ///         The iteration number.
    std::size_t iteration() const;

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

  protected:

    /// Convergence test. It may be overloaded using virtual inheritance and
    /// this base criterion may be called from derived, both in C++ and Python.
    ///
    /// *Arguments*
    ///     r (_GenericVector_)
    ///         Residual for criterion evaluation.
    ///     nonlinear_problem (_NonlinearProblem_)
    ///         The nonlinear problem.
    ///     iteration (std::size_t)
    ///         Newton iteration number.
    ///
    /// *Returns*
    ///     bool
    ///         Whether convergence occurred.
    virtual bool converged(const GenericVector& r,
                           const NonlinearProblem& nonlinear_problem,
                           std::size_t iteration);

    // FIXME: Think through carefully whether to use references or shared ptrs.
    //        If nonlinear_problem used shared_ptrs, there would be no need
    //        to pass A, P here. Note also that references are problematic in
    //        Python (the same problem in NonlinearProblem). (We could check
    //        in directorin wrappers that Python refcount is not increased.)

    /// Setup linear solver to be used with system matrix A and preconditioner
    /// matrix P. It may be overloaded to get finer control over linear solver
    /// setup.
    ///
    /// *Arguments*
    ///     linear_solver (_GenericLinearSolver)
    ///         Linear solver used for upcoming Newton step.
    ///     A (_std::shared_ptr<const GenericMatrix>_)
    ///         System Jacobian matrix.
    ///     J (_std::shared_ptr<const GenericMatrix>_)
    ///         System preconditioner matrix.
    ///     nonlinear_problem (_NonlinearProblem_)
    ///         The nonlinear problem.
    ///     iteration (std::size_t)
    ///         Newton iteration number.
    virtual void linear_solver_setup(GenericLinearSolver& linear_solver,
                                     std::shared_ptr<const GenericMatrix> A,
                                     std::shared_ptr<const GenericMatrix> P,
                                     const NonlinearProblem& nonlinear_problem,
                                     std::size_t interation);

  private:

    // Current number of Newton iterations
    std::size_t _newton_iteration;

    // Most recent residual and initial residual
    double _residual, _residual0;

    // Solver
    std::shared_ptr<GenericLinearSolver> _solver;

    // Jacobian matrix
    std::shared_ptr<GenericMatrix> _matA;

    // Preconditioner matrix
    std::shared_ptr<GenericMatrix> _matP;

    // Solution vector
    std::shared_ptr<GenericVector> _dx;

    // Residual vector
    std::shared_ptr<GenericVector> _b;

    // MPI communicator
    MPI_Comm _mpi_comm;

  };

}

#endif
