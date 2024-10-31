// Copyright (C) 2005-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

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
/// @brief A Newton solver for nonlinear systems of equations of the
/// form \f$F(x) = 0\f$.
///
/// It solves \f[ \left. \frac{dF}{dx} \right|_{x} \Delta x = F(x) \f]
/// with default update \f$x \leftarrow x - \Delta x\f$.
///
/// It relies on PETSc for linear algebra backends.
class NewtonSolver
{
public:
  /// @brief Create nonlinear solver
  /// @param[in] comm MPI communicator for the solver
  explicit NewtonSolver(MPI_Comm comm);

  /// Move constructor
  NewtonSolver(NewtonSolver&& solver) = default;

  // Copy constructor (deleted)
  NewtonSolver(const NewtonSolver& solver) = delete;

  /// Move assignment constructor
  NewtonSolver& operator=(NewtonSolver&& solver) = default;

  // Assignment operator (deleted)
  NewtonSolver& operator=(const NewtonSolver& solver) = delete;

  /// Destructor
  ~NewtonSolver();

  /// @brief Set the function for computing the residual \f$F(x) = 0\f$
  /// and the vector to assemble the residual into.
  /// @param[in] F Function to compute/assemble the residual vector `b`.
  /// The first argument to the function is the solution vector `x` and
  /// the second is the vector `b` to assemble into.
  /// @param[in] b Vector to assemble to residual into.
  void setF(std::function<void(const Vec, Vec)> F, Vec b);

  /// @brief Set the function for computing the Jacobian \f$J:=dF/dx\f$
  /// and the matrix to assemble the Jacobian into.
  /// @param[in] J Function to compute the Jacobian matrix `Jmat`. The
  /// first argument to the function is the solution vector `x` and the
  /// second is the matrix to assemble into.
  /// @param[in] Jmat Matrix to assemble the Jacobian into.
  void setJ(std::function<void(const Vec, Mat)> J, Mat Jmat);

  /// @brief Set the function for computing the preconditioner matrix.
  ///
  /// It is optional to set the preconditioner matrix. By default the
  /// solver will use the Jacobian matrix.
  ///
  /// @param[in] P Function to compute the preconditioner matrix `Pmat`.
  /// The first argument to the function is the solution vector `x` and
  /// the second is the matrix to assemble into.
  /// @param[in] Pmat Matrix to assemble the preconditioner into.
  void setP(std::function<void(const Vec, Mat)> P, Mat Pmat);

  /// @brief Get the internal Krylov solver used to solve for the Newton
  /// updates (const version).
  ///
  /// The Krylov solver prefix is `nls_solve_`.
  ///
  /// @return The Krylov solver
  const la::petsc::KrylovSolver& get_krylov_solver() const;

  /// @brief Get the internal Krylov solver used to solve for the Newton
  /// updates (non-const version).
  ///
  /// The Krylov solver prefix is `nls_solve_`.
  ///
  /// @return The Krylov solver
  la::petsc::KrylovSolver& get_krylov_solver();

  /// @brief Set the function that is called before the residual or
  /// Jacobian are computed. It is commonly used to update ghost values.
  ///
  /// @param[in] form Function to call. It takes the (latest) solution
  /// vector `x` as an argument.
  void set_form(std::function<void(Vec)> form);

  /// @brief Set function that is called at the end of each Newton
  /// iteration to test for convergence.
  /// @param[in] c Function that tests for convergence
  void set_convergence_check(
      std::function<std::pair<double, bool>(const NewtonSolver&, const Vec)> c);

  /// @brief Optional set function that is called after each inner solve for
  /// the Newton increment to update the solution.
  ///
  /// The function `update` takes `this`, the Newton increment `dx`, and
  /// the vector `x` from the start of the Newton iteration.
  ///
  /// By default, the update is x <- x - dx
  ///
  /// @param[in] update The function that updates the solution
  void set_update(
      std::function<void(const NewtonSolver& solver, const Vec, Vec)> update);

  /// @brief Solve the nonlinear problem.
  ///
  /// @param[in,out] x The solution vector. It should be set the initial
  /// solution guess.
  /// @return (number of Newton iterations, whether iteration converged)
  std::pair<int, bool> solve(Vec x);

  /// @brief Get number of Newton iterations. It can can be called by
  /// functions that check for convergence during a solve.
  /// @return Number of Newton iterations performed.
  int iteration() const;

  /// @brief Number of Krylov iterations elapsed since solve started.
  /// @return Number of Krylov iterations.
  int krylov_iterations() const;

  /// @brief Get current residual.
  /// @return Current residual.
  double residual() const;

  /// @brief Get initial residual.
  /// @return Initial residual.
  double residual0() const;

  /// @brief Get MPI communicator.
  MPI_Comm comm() const;

  /// @brief Maximum number of iterations.
  int max_it = 50;

  /// @brief Relative convergence tolerance.
  double rtol = 1e-9;

  /// @brief Absolute convergence tolerance.
  double atol = 1e-10;

  /// @todo change to string to enum.
  /// @brief Convergence criterion.
  std::string convergence_criterion = "residual";

  /// @brief Monitor convergence.
  bool report = true;

  /// @brief Throw error if solver fails to converge.
  bool error_on_nonconvergence = true;

  /// @brief Relaxation parameter.
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

  // Function to update the solution after inner solve for the Newton increment
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

#endif
