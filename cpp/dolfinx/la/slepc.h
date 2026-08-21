// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_SLEPC

#include <complex>
#include <dolfinx/common/MPI.h>
#include <petscmat.h>
#include <petscvec.h>
#include <slepceps.h>
#include <string>
#include <string_view>

namespace dolfinx::la
{

/// @brief This class provides an eigenvalue solver for PETSc matrices.
/// It is a wrapper for the SLEPc eigenvalue solver.
class SLEPcEigenSolver
{
public:
  /// @brief Create eigenvalue solver.
  /// @note SLEPc must have been initialised (`SlepcInitialize`) before
  /// this is called.
  /// @param[in] comm MPI communicator.
  explicit SLEPcEigenSolver(MPI_Comm comm);

  /// @brief Create eigenvalue solver from an existing EPS object.
  /// @param[in] eps SLEPc EPS object, which must already have been
  /// created. The reference count of `eps` is always decreased when
  /// this SLEPcEigenSolver is destroyed.
  /// @param[in] inc_ref_count True if the reference count of `eps`
  /// should be incremented.
  SLEPcEigenSolver(EPS eps, bool inc_ref_count);

  // Copy constructor (deleted)
  SLEPcEigenSolver(const SLEPcEigenSolver&) = delete;

  /// Move constructor
  SLEPcEigenSolver(SLEPcEigenSolver&& solver) noexcept;

  /// Destructor
  ~SLEPcEigenSolver();

  // Assignment operator (deleted)
  SLEPcEigenSolver& operator=(const SLEPcEigenSolver&) = delete;

  /// Move assignment
  SLEPcEigenSolver& operator=(SLEPcEigenSolver&& solver) noexcept;

  /// @brief Set the operators defining the eigenvalue problem.
  /// @param[in] A Operator for the standard problem
  /// \f$A x = \lambda x\f$, or the left-hand operator of the
  /// generalised problem \f$A x = \lambda B x\f$. Must not be null.
  /// @param[in] B Right-hand operator of the generalised problem, or
  /// `nullptr` for a standard eigenvalue problem.
  void set_operators(const Mat A, const Mat B);

  /// @brief Solve the eigenvalue problem \f$A x = \lambda x\f$ (or
  /// \f$A x = \lambda B x\f$).
  /// @note The number of eigenpairs to compute, and every other solver
  /// setting, is applied to the EPS object returned by eps(). SLEPc
  /// sizes its working subspace from the requested number, which
  /// defaults to one, e.g.
  /// @code
  /// SLEPcEigenSolver solver(comm);
  /// solver.set_operators(A, nullptr);
  /// EPSSetDimensions(solver.eps(), 5, PETSC_DETERMINE, PETSC_DETERMINE);
  /// EPSSetWhichEigenpairs(solver.eps(), EPS_SMALLEST_REAL);
  /// solver.solve();
  /// @endcode
  void solve();

  /// @brief Get the ith eigenvalue.
  /// @param[in] i Index of the eigenvalue, in `[0, nconv)`, where
  /// `nconv` is the number of converged eigenpairs reported by
  /// `EPSGetConverged(eps(), &nconv)`.
  /// @return The eigenvalue.
  std::complex<PetscReal> get_eigenvalue(PetscInt i) const;

  /// @brief Get the ith eigenpair.
  /// @param[out] lr Real part of the eigenvalue.
  /// @param[out] lc Imaginary part of the eigenvalue.
  /// @param[out] r Real part of the eigenvector.
  /// @param[out] c Imaginary part of the eigenvector.
  /// @param[in] i Index of the eigenpair, in `[0, nconv)`, where
  /// `nconv` is the number of converged eigenpairs reported by
  /// `EPSGetConverged(eps(), &nconv)`.
  /// @note For a complex PETSc scalar type the eigenvalue and
  /// eigenvector are held entirely in `lr` and `r`, and `lc` and `c`
  /// are set to zero.
  void get_eigenpair(PetscScalar& lr, PetscScalar& lc, Vec r, Vec c,
                     PetscInt i) const;

  /// Sets the prefix used by PETSc when searching the PETSc options
  /// database
  void set_options_prefix(std::string_view options_prefix);

  /// Returns the prefix used by PETSc when searching the PETSc options
  /// database
  std::string get_options_prefix() const;

  /// @brief Set options from the PETSc options database.
  /// @note Call after set_options_prefix and after any settings the
  /// database should override.
  void set_from_options() const;

  /// Return SLEPc EPS pointer
  EPS eps() const;

  /// Return MPI communicator
  MPI_Comm comm() const;

private:
  // SLEPc solver pointer
  EPS _eps;
};
} // namespace dolfinx::la
#endif
