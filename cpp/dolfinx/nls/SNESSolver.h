// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include <exception>
#include <functional>
#include <mpi.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscsystypes.h>
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
/// An exception thrown by a callback aborts the solve and is re-thrown
/// by solve().
///
/// Example:
/// @code
/// nls::petsc::SNESSolver solver(mesh.comm());
/// solver.set_F([&](const Vec x, Vec b) { assemble_residual(x, b); }, b);
/// solver.set_J([&](const Vec x, Mat A, Mat) { assemble_jacobian(x, A); },
///               A);
/// solver.set_options_prefix("my_problem_");
/// solver.set_from_options();
/// solver.solve(x);
/// @endcode
class SNESSolver
{
public:
  /// @brief Create a nonlinear solver.
  /// @param[in] comm MPI communicator for the solver.
  explicit SNESSolver(MPI_Comm comm);

  /// @brief Create a solver wrapper of a PETSc SNES object.
  ///
  /// @note The callbacks registered on `snes` hold a pointer to the
  /// solver, which is not reference counted. Using `snes` once the
  /// solver has been destroyed is undefined.
  ///
  /// @param[in] snes PETSc SNES object. It should already have been
  /// created.
  /// @param[in] inc_ref_count Increment the reference count on `snes`
  /// if true, so that it outlives a caller that destroys their own
  /// reference.
  SNESSolver(SNES snes, bool inc_ref_count);

  // Copy constructor (deleted)
  SNESSolver(const SNESSolver& solver) = delete;

  /// Move constructor
  /// @note The `SNES` callback context is a pointer to the owning
  /// solver, so moving re-registers the callbacks.
  SNESSolver(SNESSolver&& solver) noexcept;

  /// Destructor
  ~SNESSolver();

  // Copy assignment (deleted)
  SNESSolver& operator=(const SNESSolver& solver) = delete;

  /// Move assignment
  SNESSolver& operator=(SNESSolver&& solver) noexcept;

  /// @brief Set the function for computing the residual \f$F(x)\f$, and
  /// the vector that defines its layout.
  ///
  /// A residual defined by a fem::Form can be assembled with
  /// fem::petsc::assemble_residual.
  ///
  /// @note `F` must assemble into the vector it is passed, which is not
  /// always `b`. A line search, for instance, evaluates the residual in
  /// a work vector duplicated from `b`.
  ///
  /// @param[in] F Function to assemble the residual at `x` into the
  /// vector it is passed. It is responsible for zeroing that vector,
  /// and for any required ghost update of `x`.
  /// @param[in] b Vector that the solver duplicates to create the
  /// vectors passed to `F`. A reference is held, so the caller can
  /// destroy their own reference.
  void set_F(std::function<void(const Vec x, Vec b)> F, Vec b);

  /// @brief Set the function for computing the Jacobian
  /// \f$J := dF/dx\f$, and the matrices that define its layout.
  ///
  /// A Jacobian defined by a fem::Form can be assembled with
  /// fem::petsc::assemble_jacobian.
  ///
  /// @note As for set_F, `J` must assemble into the matrices it is
  /// passed rather than into `Jmat` and `Pmat` directly.
  ///
  /// @param[in] J Function to assemble the Jacobian at `x` into the
  /// first matrix it is passed and the preconditioner into the second.
  /// It is responsible for zeroing them and for finalising assembly.
  /// @param[in] Jmat Matrix defining the layout of the Jacobian.
  /// @param[in] Pmat Matrix defining the layout of the preconditioner.
  /// If `nullptr`, `Jmat` is used, the two matrices passed to `J` are
  /// the same, and `J` should assemble the Jacobian only. A reference
  /// to each matrix is held, so the caller can destroy their own
  /// references.
  void set_J(std::function<void(const Vec x, Mat Jmat, Mat Pmat)> J, Mat Jmat,
             Mat Pmat = nullptr);

  /// @brief Set a function called before each nonlinear iteration, e.g.
  /// to update a time- or step-dependent term.
  /// @param[in] update Function called with the index of the iteration
  /// that is about to be taken.
  void set_update(std::function<void(int step)> update);

  /// @brief Solve \f$F(x) = 0\f$.
  ///
  /// Non-convergence is not treated as an error (a warning is logged);
  /// use snes() and `SNESGetConvergedReason`, or the
  /// `-snes_error_if_not_converged` option, if it matters.
  ///
  /// @note An exception thrown by a callback is re-thrown here. It must
  /// be thrown on all ranks, otherwise ranks that continue the solve
  /// deadlock on a collective operation. PETSc unwinds the aborted
  /// solve without restoring its state, e.g. `x` is left locked for
  /// read-only access, so neither this solver nor the vectors and
  /// matrices it holds can be used again.
  ///
  /// @param[in,out] x Solution vector, holding the initial guess on
  /// entry.
  /// @return Number of nonlinear iterations performed.
  int solve(Vec x);

  /// @brief Set the prefix used by PETSc when searching the PETSc
  /// options database.
  /// @param[in] options_prefix Prefix to set. Conventionally ends with
  /// `_`.
  void set_options_prefix(std::string_view options_prefix);

  /// @brief Get the prefix used by PETSc when searching the PETSc
  /// options database.
  /// @return The options prefix.
  std::string get_options_prefix() const;

  /// @brief Set options from the PETSc options database.
  void set_from_options() const;

  /// @brief Get the wrapped PETSc SNES object, e.g. to configure the
  /// line search or the Krylov solver used for each iteration.
  /// @return The PETSc SNES object. The solver retains ownership.
  SNES snes() const;

private:
  // Callbacks passed to SNESSetFunction/SNESSetJacobian. The context
  // pointer is the owning SNESSolver.
  static PetscErrorCode residual(SNES snes, Vec x, Vec b, void* ctx);
  static PetscErrorCode jacobian(SNES snes, Vec x, Mat Jmat, Mat Pmat,
                                 void* ctx);

  // Callback passed to SNESSetUpdate. It takes no context argument, so
  // the owning SNESSolver is recovered from the SNES object.
  static PetscErrorCode update_step(SNES snes, PetscInt step);

  // Store an in-flight exception and stop the solve. Called by the
  // callbacks, from where an exception cannot be propagated through the
  // PETSc C frames.
  PetscErrorCode store_exception();

  // Register the callbacks that have been set, and attach this as the
  // context that the callbacks recover the solver from
  void set_callbacks();

  // Function for computing the residual vector
  std::function<void(const Vec x, Vec b)> _fnF;

  // Function for computing the Jacobian and preconditioner matrices
  std::function<void(const Vec x, Mat Jmat, Mat Pmat)> _fnJ;

  // Function called before each nonlinear iteration
  std::function<void(int step)> _fnupdate;

  // Exception thrown by a callback during a solve, re-thrown by solve
  std::exception_ptr _exception;

  // Residual vector
  Vec _b = nullptr;

  // Jacobian and preconditioner matrices
  Mat _matJ = nullptr, _matP = nullptr;

  // PETSc solver pointer
  SNES _snes;
};
} // namespace dolfinx::nls::petsc

#endif
