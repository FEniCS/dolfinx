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
  /// @param[in] comm The MPI communicator for the solver
  explicit NewtonSolver(MPI_Comm comm);

  /// Destructor
  virtual ~NewtonSolver();

  /// Solve abstract nonlinear problem \f$`F(x) = 0\f$ for given \f$F\f$
  /// and Jacobian \f$\dfrac{\partial F}{\partial x}\f$.
  ///
  /// @param[in] nonlinear_function The nonlinear problem
  /// @param[in,out] x The vector
  /// @return Pair of number of Newton iterations, and whether iteration
  ///         converged)
  std::pair<int, bool> solve(NonlinearProblem& nonlinear_function, Vec x);

  /// Return number of Krylov iterations elapsed since
  /// solve started
  /// @return Number of iterations.
  int krylov_iterations() const;

  /// Return current residual
  /// @return Current residual
  double residual() const;

  /// Return initial residual
  /// @returns Initial residual
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
  /// Convergence test. It may be overloaded using virtual inheritance
  /// and this base criterion may be called from derived, both in C++
  /// and Python.
  ///
  /// @param r Residual for criterion evaluation
  /// @param nonlinear_problem The nonlinear problem
  /// @param iteration Newton iteration number
  /// @returns  True if convergence achieved
  virtual bool converged(const Vec r, const NonlinearProblem& nonlinear_problem,
                         std::size_t iteration);

  /// Update solution vector by computed Newton step. Default update is
  /// given by formula::
  ///
  ///   x -= relaxation_parameter*dx
  ///
  ///  @param[in,out] x The solution vector to be updated
  ///  @param dx The update vector computed by Newton step
  ///  @param[in] relaxation_parameter Newton relaxation parameter
  ///  @param nonlinear_problem The nonlinear problem
  ///  @param[in] iteration Newton iteration number
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
