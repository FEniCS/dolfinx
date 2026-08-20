// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include <functional>
#include <mpi.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>
#include <string>
#include <string_view>

namespace dolfinx::nls::petsc
{
/// @brief Solver for nonlinear systems \f$F(x) = 0\f$ using PETSc SNES.
///
/// Adapts C++ callables to the SNES callback interface and holds a
/// reference to the assembled-into vector and matrices. Configuration
/// of the solve is left to the user, via the options database or the
/// `SNES` object returned by snes().
///
/// Example:
/// @code
/// nls::petsc::NonlinearProblem problem(mesh.comm());
/// problem.set_F([&](const Vec x, Vec b) { assemble_residual(x, b); }, b);
/// problem.set_J([&](const Vec x, Mat A, Mat) { assemble_jacobian(x, A); },
///               A);
/// problem.set_options_prefix("my_problem_");
/// problem.set_from_options();
/// problem.solve(x);
/// @endcode
class NonlinearProblem
{
public:
  /// @brief Create a nonlinear solver.
  /// @param[in] comm MPI communicator for the solver.
  explicit NonlinearProblem(MPI_Comm comm);

  /// @brief Create a solver wrapper of a PETSc SNES object.
  /// @param[in] snes PETSc SNES object. It should already have been
  /// created.
  /// @param[in] inc_ref_count Increment the reference count on `snes`
  /// if true.
  NonlinearProblem(SNES snes, bool inc_ref_count);

  // Copy constructor (deleted)
  NonlinearProblem(const NonlinearProblem& problem) = delete;

  /// Move constructor
  /// @note The `SNES` callback context is a pointer to the owning
  /// problem, so moving re-registers the callbacks.
  NonlinearProblem(NonlinearProblem&& problem) noexcept;

  /// Destructor
  ~NonlinearProblem();

  // Copy assignment (deleted)
  NonlinearProblem& operator=(const NonlinearProblem& problem) = delete;

  /// Move assignment
  NonlinearProblem& operator=(NonlinearProblem&& problem) noexcept;

  /// @brief Set the function for computing the residual \f$F(x)\f$ and
  /// the vector to assemble it into.
  /// @param[in] F Function to assemble the residual at `x` into `b`. It
  /// is responsible for any required ghost update of `x`.
  /// @param[in] b Vector to assemble the residual into. A reference is
  /// held, so the caller can destroy their own reference.
  void set_F(std::function<void(const Vec x, Vec b)> F, Vec b);

  /// @brief Set the function for computing the Jacobian
  /// \f$J := dF/dx\f$ and the matrices to assemble it into.
  /// @param[in] J Function to assemble the Jacobian at `x` into `Jmat`
  /// and the preconditioner into `Pmat`. It is responsible for
  /// finalising assembly.
  /// @param[in] Jmat Matrix to assemble the Jacobian into.
  /// @param[in] Pmat Matrix to assemble the preconditioner into. If
  /// `nullptr`, `Jmat` is used, and `J` should assemble into `Jmat`
  /// only. A reference to each matrix is held, so the caller can
  /// destroy their own references.
  void set_J(std::function<void(const Vec x, Mat Jmat, Mat Pmat)> J, Mat Jmat,
             Mat Pmat = nullptr);

  /// @brief Solve \f$F(x) = 0\f$.
  ///
  /// Non-convergence is not treated as an error (a warning is logged);
  /// use snes() and `SNESGetConvergedReason`, or the
  /// `-snes_error_if_not_converged` option, if it matters.
  ///
  /// @param[in,out] x Solution vector, holding the initial guess on
  /// entry.
  /// @return Number of nonlinear iterations performed.
  int solve(Vec x) const;

  /// Sets the prefix used by PETSc when searching the PETSc options
  /// database
  void set_options_prefix(std::string_view options_prefix);

  /// Returns the prefix used by PETSc when searching the PETSc options
  /// database
  std::string get_options_prefix() const;

  /// Set options from PETSc options database
  void set_from_options() const;

  /// Return PETSc SNES pointer
  SNES snes() const;

private:
  // Callbacks passed to SNESSetFunction/SNESSetJacobian. The context
  // pointer is the owning NonlinearProblem.
  static PetscErrorCode residual(SNES snes, Vec x, Vec b, void* ctx);
  static PetscErrorCode jacobian(SNES snes, Vec x, Mat Jmat, Mat Pmat,
                                 void* ctx);

  // Register the callbacks that have been set, with this as context
  void set_callbacks();

  // Function for computing the residual vector
  std::function<void(const Vec x, Vec b)> _fnF;

  // Function for computing the Jacobian and preconditioner matrices
  std::function<void(const Vec x, Mat Jmat, Mat Pmat)> _fnJ;

  // Residual vector
  Vec _b = nullptr;

  // Jacobian and preconditioner matrices
  Mat _matJ = nullptr, _matP = nullptr;

  // PETSc solver pointer
  SNES _snes;
};
} // namespace dolfinx::nls::petsc

#endif
