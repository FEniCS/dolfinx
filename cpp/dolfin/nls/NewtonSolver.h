// Copyright (C) 2005-2008 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include <memory>
#include <utility>

namespace dolfin
{

namespace la
{
class PETScMatrix;
class PETScVector;
class PETScKrylovSolver;
}

namespace nls
{

class NonlinearProblem;

/// This class defines a Newton solver for nonlinear systems of
/// equations of the form \f$F(x) = 0\f$.

class NewtonSolver : public common::Variable
{
public:
  /// Create nonlinear solver
  /// @param comm (MPI_Comm)
  explicit NewtonSolver(MPI_Comm comm);

  /// Destructor
  virtual ~NewtonSolver();

  /// Solve abstract nonlinear problem \f$`F(x) = 0\f$ for given
  /// \f$F\f$ and Jacobian \f$\dfrac{\partial F}{\partial x}\f$.
  ///
  /// @param    nonlinear_function (_NonlinearProblem_)
  ///         The nonlinear problem.
  /// @param    x (_la::PETScVector_)
  ///         The vector.
  ///
  /// @returns    std::pair<std::size_t, bool>
  ///         Pair of number of Newton iterations, and whether
  ///         iteration converged)
  std::pair<std::size_t, bool> solve(NonlinearProblem& nonlinear_function,
                                     la::PETScVector& x);

  /// Return current Newton iteration number
  ///
  /// @returns     std::size_t
  ///         The iteration number.
  std::size_t iteration() const;

  /// Return number of Krylov iterations elapsed since
  /// solve started
  ///
  /// @returns    std::size_t
  ///         The number of iterations.
  std::size_t krylov_iterations() const;

  /// Return current residual
  ///
  /// @returns double
  ///         Current residual.
  double residual() const;

  /// Return initial residual
  ///
  /// @returns double
  ///         Initial residual.
  double residual0() const;

  /// Return current relative residual
  ///
  /// @returns double
  ///       Current relative residual.
  double relative_residual() const;

  /// Default parameter values
  ///
  /// @returns _Parameters_
  ///         Parameter values.
  static parameter::Parameters default_parameters();

  /// Set relaxation parameter. Default value 1.0 means full
  /// Newton method, value smaller than 1.0 relaxes the method
  /// by shrinking effective Newton step size by the given factor.
  ///
  /// @param relaxation_parameter (double)
  ///         Relaxation parameter value.
  void set_relaxation_parameter(double relaxation_parameter)
  {
    _relaxation_parameter = relaxation_parameter;
  }

  /// Get relaxation parameter
  ///
  /// @returns    double
  ///         Relaxation parameter value.
  double get_relaxation_parameter() { return _relaxation_parameter; }

protected:
  /// Convergence test. It may be overloaded using virtual inheritance and
  /// this base criterion may be called from derived, both in C++ and Python.
  ///
  /// @param r (_la::PETScVector_)
  ///         Residual for criterion evaluation.
  /// @param nonlinear_problem (_NonlinearProblem_)
  ///         The nonlinear problem.
  /// @param iteration (std::size_t)
  ///         Newton iteration number.
  ///
  /// @returns  bool
  ///         Whether convergence occurred.
  virtual bool converged(const la::PETScVector& r,
                         const NonlinearProblem& nonlinear_problem,
                         std::size_t iteration);

  /// Setup solver to be used with system matrix A and preconditioner
  /// matrix P. It may be overloaded to get finer control over linear
  /// solver setup, various linesearch tricks, etc. Note that minimal
  /// implementation should call *set_operators* method of the linear
  /// solver.
  ///
  /// @param A (_std::shared_ptr<const PETScMatrix>_)
  ///         System Jacobian matrix.
  /// @param J (_std::shared_ptr<const PETSccMatrix>_)
  ///         System preconditioner matrix.
  /// @param nonlinear_problem (_NonlinearProblem_)
  ///         The nonlinear problem.
  /// @param iteration (std::size_t)
  ///         Newton iteration number.
  virtual void solver_setup(std::shared_ptr<const la::PETScMatrix> A,
                            std::shared_ptr<const la::PETScMatrix> P,
                            const NonlinearProblem& nonlinear_problem,
                            std::size_t iteration);

  /// Update solution vector by computed Newton step. Default
  /// update is given by formula::
  ///
  ///   x -= relaxation_parameter*dx
  ///
  ///  @param x (_la::PETScVector>_)
  ///         The solution vector to be updated.
  ///  @param dx (_la::PETScVector>_)
  ///         The update vector computed by Newton step.
  ///  @param relaxation_parameter (double)
  ///         Newton relaxation parameter.
  ///  @param nonlinear_problem (_NonlinearProblem_)
  ///         The nonlinear problem.
  ///  @param iteration (std::size_t)
  ///         Newton iteration number.
  virtual void update_solution(la::PETScVector& x, const la::PETScVector& dx,
                               double relaxation_parameter,
                               const NonlinearProblem& nonlinear_problem,
                               std::size_t iteration);

private:
  // Current number of Newton iterations
  std::size_t _newton_iteration;

  // Accumulated number of Krylov iterations since solve began
  std::size_t _krylov_iterations;

  // Relaxation parameter
  double _relaxation_parameter;

  // Most recent residual and initial residual
  double _residual, _residual0;

  // Solver
  std::shared_ptr<la::PETScKrylovSolver> _solver;

  // Jacobian matrix
  std::shared_ptr<la::PETScMatrix> _matA;

  // Preconditioner matrix
  std::shared_ptr<la::PETScMatrix> _matP;

  // Solution vector
  std::shared_ptr<la::PETScVector> _dx;

  // Residual vector
  std::shared_ptr<la::PETScVector> _b;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;
};
}
}
