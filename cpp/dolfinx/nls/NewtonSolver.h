// Copyright (C) 2005-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <dolfinx/la/petsc.h>
#include <functional>
#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <utility>

namespace dolfinx
{

namespace la::petsc
{
class KrylovSolver;
} // namespace la::petsc

namespace nls::petsc
{

/// This class defines a Newton solver for nonlinear systems of
/// equations of the form \f$F(x) = 0\f$.

class NewtonSolver
{
public:
  /// Create nonlinear solver
  /// @param[in] comm The MPI communicator for the solver
  explicit NewtonSolver(MPI_Comm comm);

  // Move constructor (deleted)
  NewtonSolver(NewtonSolver&& solver) = delete;

  // Copy constructor (deleted)
  NewtonSolver(const NewtonSolver& solver) = delete;

  // Assignment operator (deleted)
  NewtonSolver& operator=(const NewtonSolver& solver) = delete;

  // Move assignment constructor (deleted)
  NewtonSolver& operator=(const NewtonSolver&& solver) = delete;

  /// Destructor
  ~NewtonSolver();

  /// Set the function for computing the residual and the vector to the
  /// assemble the residual into
  /// @param[in] F Function to compute the residual vector b (x, b)
  /// @param[in] b The vector to assemble to residual into
  void setF(const std::function<void(const Vec, Vec)>& F, Vec b);

  /// Set the function for computing the Jacobian (dF/dx) and the matrix
  /// to assemble the residual into
  /// @param[in] J Function to compute the Jacobian matrix b (x, A)
  /// @param[in] Jmat The matrix to assemble the Jacobian into
  void setJ(const std::function<void(const Vec, Mat)>& J, Mat Jmat);

  /// Set the function for computing the preconditioner matrix (optional)
  /// @param[in] P Function to compute the preconditioner matrix b (x, P)
  /// @param[in] Pmat The matrix to assemble the preconditioner into
  void setP(const std::function<void(const Vec, Mat)>& P, Mat Pmat);

  /// Get the internal Krylov solver used to solve for the Newton updates
  /// const version
  /// The Krylov solver prefix is nls_solve_
  /// @return The Krylov solver
  const la::petsc::KrylovSolver& get_krylov_solver() const;

  /// Get the internal Krylov solver used to solve for the Newton updates
  /// non-const version
  /// The Krylov solver prefix is nls_solve_
  /// @return The Krylov solver
  la::petsc::KrylovSolver& get_krylov_solver();

  /// Set the function that is called before the residual or Jacobian
  /// are computed. It is commonly used to update ghost values.
  /// @param[in] form The function to call. It takes the latest solution
  /// vector @p x as an argument
  void set_form(const std::function<void(Vec)>& form);

  /// Set function that is called at the end of each Newton iteration to
  /// test for convergence.
  /// @param[in] c The function that tests for convergence
  void set_convergence_check(const std::function<std::pair<double, bool>(
                                 const NewtonSolver&, const Vec)>& c);

  /// Set function that is called after each Newton iteration to update
  /// the solution
  /// @param[in] update The function that updates the solution
  void set_update(const std::function<void(const NewtonSolver& solver,
                                           const Vec, Vec)>& update);

  /// Solve the nonlinear problem \f$`F(x) = 0\f$ for given \f$F\f$ and
  /// Jacobian \f$\dfrac{\partial F}{\partial x}\f$.
  ///
  /// @param[in,out] x The vector
  /// @return (number of Newton iterations, whether iteration converged)
  std::pair<int, bool> solve(Vec x);

  /// The number of Newton interations. It can can called by functions
  /// that check for convergence during a solve.
  /// @return The number of Newton iterations performed
  int iteration() const;

  /// Return number of Krylov iterations elapsed since
  /// solve started
  /// @return Number of iterations.
  int krylov_iterations() const;

  /// Return current residual
  /// @return Current residual
  double residual() const;

  /// Return initial residual
  /// @return Initial residual
  double residual0() const;

  /// Return MPI communicator
  MPI_Comm comm() const;

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

private:
  // Function for computing the residual vector. The first argument is
  // the latest solution vector x and the second argument is the
  // residual vector.
  std::function<void(const Vec x, Vec b)> _fnF;

  // Function for computing the Jacobian matrix operator. The first
  // argument is the latest solution vector x and the second argument is
  // the matrix operator.
  std::function<void(const Vec x, Mat J)> _fnJ;

  // Function for computing the preconditioner matrix operator. The
  // first argument is the latest solution vector x and the second
  // argument is the matrix operator.
  std::function<void(const Vec x, Mat P)> _fnP;

  // Function called before the residual and Jacobian function at each
  // iteration.
  std::function<void(const Vec x)> _system;

  // Residual vector
  Vec _b = nullptr;

  // Jacobian matrix and preconditioner matrix
  Mat _matJ = nullptr, _matP = nullptr;

  // Function to check for convergence
  std::function<std::pair<double, bool>(const NewtonSolver& solver,
                                        const Vec r)>
      _converged;

  // Function to update the solution once convergence is reached
  std::function<void(const NewtonSolver& solver, const Vec dx, Vec x)>
      _update_solution;

  // Accumulated number of Krylov iterations since solve began
  int _krylov_iterations;

  // Number of iterations
  int _iteration;

  // Most recent residual and initial residual
  double _residual, _residual0;

  // Linear solver
  la::petsc::KrylovSolver _solver;

  // Solution vector
  Vec _dx = nullptr;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};
} // namespace nls::petsc
} // namespace dolfinx
