// Copyright (C) 2005-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <memory>
#include <petscvec.h>
#include <utility>

namespace dolfin
{

namespace la
{
class PETScKrylovSolver;
} // namespace la

namespace nls
{

class NonlinearProblem;

/// This class defines a Newton solver for nonlinear systems of
/// equations of the form \f$F(x) = 0\f$.

class NewtonSolver
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
  /// @param    x (_Vec_)
  ///         The vector.
  ///
  /// @returns    std::pair<std::size_t, bool>
  ///         Pair of number of Newton iterations, and whether
  ///         iteration converged)
  std::pair<int, bool> solve(NonlinearProblem& nonlinear_function, Vec x);

  /// Return number of Krylov iterations elapsed since
  /// solve started
  ///
  /// @returns    std::size_t
  ///         The number of iterations.
  int krylov_iterations() const;

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

  /// Maximum number of iterations
  int max_it = 50;

  /// Relative tolerance
  double rtol = 1e-9;

  /// Absolute tolerance
  double atol = 1e-10;

  // FIXME: change to string to enum
  /// Convergence criterion
  std::string convergence_criterion = "residual";

  /// Monitor convergence
  bool report = true;

  /// Throw error if solver fails to converge
  bool error_on_nonconvergence = true;

  /// Relaxation parameter
  double relaxation_parameter = 1.0;

protected:
  /// Convergence test. It may be overloaded using virtual inheritance and
  /// this base criterion may be called from derived, both in C++ and Python.
  ///
  /// @param r (_Vec_)
  ///         Residual for criterion evaluation.
  /// @param nonlinear_problem (_NonlinearProblem_)
  ///         The nonlinear problem.
  /// @param iteration (std::size_t)
  ///         Newton iteration number.
  ///
  /// @returns  bool
  ///         Whether convergence occurred.
  virtual bool converged(const Vec r, const NonlinearProblem& nonlinear_problem,
                         std::size_t iteration);

  /// Update solution vector by computed Newton step. Default
  /// update is given by formula::
  ///
  ///   x -= relaxation_parameter*dx
  ///
  ///  @param x (_Vec_)
  ///         The solution vector to be updated.
  ///  @param dx (_Vec_)
  ///         The update vector computed by Newton step.
  ///  @param relaxation_parameter (double)
  ///         Newton relaxation parameter.
  ///  @param nonlinear_problem (_NonlinearProblem_)
  ///         The nonlinear problem.
  ///  @param iteration (std::size_t)
  ///         Newton iteration number.
  virtual void update_solution(Vec x, const Vec dx, double relaxation_parameter,
                               const NonlinearProblem& nonlinear_problem,
                               std::size_t iteration);

private:
  // Accumulated number of Krylov iterations since solve began
  int _krylov_iterations;

  // Most recent residual and initial residual
  double _residual, _residual0;

  // Solver
  la::PETScKrylovSolver _solver;

  // Solution vector
  Vec _dx;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;
};
} // namespace nls
} // namespace dolfin
