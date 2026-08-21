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
/// Adapts C++ callables to the SNES callback interface and handles
/// memory management. Configuration of the solve is left to the user,
/// via the options database or the `SNES` object returned by snes().
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
  /// @note `F` must assemble into the `b` it is passed, which is not
  /// always `b_layout`. A line search, for instance, evaluates the
  /// residual in a work vector duplicated from `b_layout`.
  ///
  /// @param[in] F Function to assemble the residual at `x` into the `b`
  /// it is passed. It is responsible for zeroing that vector, and for
  /// any required ghost update of `x`.
  /// @param[in] b_layout Vector that the solver may duplicate to create
  /// the vectors passed to `F`. A reference is held, so the caller can
  /// destroy their own reference.
  void set_F(std::function<void(const Vec x, Vec b)> F, Vec b_layout);

  /// @brief Set the function for computing the Jacobian
  /// \f$J := dF/dx\f$, and the matrices that define its layout.
  ///
  /// A Jacobian defined by a fem::Form can be assembled with
  /// fem::petsc::assemble_jacobian.
  ///
  /// @note `J` must assemble into the `Jmat` and `Pmat` it is passed,
  /// which are not always `J_layout` and `P_layout`.
  ///
  /// @param[in] J Function to assemble the Jacobian at `x` into the
  /// `Jmat` it is passed, and the preconditioner into the `Pmat` it is
  /// passed. It is responsible for zeroing the matrices it assembles
  /// into and for finalising assembly.
  /// @param[in] J_layout Matrix defining the layout of the Jacobian.
  /// @param[in] P_layout Matrix defining the layout of the
  /// preconditioner. If `nullptr`, `J_layout` is used, the two matrices
  /// passed to `J` are the same, and `J` should assemble the Jacobian
  /// only. A reference to each matrix is held, so the caller can
  /// destroy their own references.
  void set_J(std::function<void(const Vec x, Mat Jmat, Mat Pmat)> J,
             Mat J_layout, Mat P_layout = nullptr);

  /// @brief Set a function called before each nonlinear iteration, e.g.
  /// to update a time- or step-dependent term.
  /// @param[in] update Function called with the index of the iteration
  /// that is about to be taken.
  void set_update(std::function<void(PetscInt step)> update);

  /// @brief Solve \f$F(x) = 0\f$.
  ///
  /// Non-convergence is not treated as an error (a warning is logged);
  /// use snes() and `SNESGetConvergedReason`, or the
  /// `-snes_error_if_not_converged` option.
  ///
  /// @note An exception thrown by a callback is re-thrown here. The
  /// callback must throw on all ranks. A solver that throws an
  /// exception must not be used again.
  ///
  /// @param[in,out] x Solution vector, holding the initial guess on
  /// entry.
  /// @return Number of nonlinear iterations performed.
  PetscInt solve(Vec x);

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
  // the owning SNESSolver is recovered from the context that set_F
  // registered on the SNES object.
  static PetscErrorCode update_step(SNES snes, PetscInt step);

  // Store an in-flight exception and stop the solve. Called by the
  // callbacks, because C++ exceptions cannot be propagated through the
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
  std::function<void(PetscInt step)> _fnupdate;

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
