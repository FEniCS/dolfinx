// Copyright (C) 2004-2015 Johan Jansson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <string>

namespace dolfinx::la
{

/// This class implements Krylov methods for linear systems of the form
/// Ax = b. It is a wrapper for the Krylov solvers of PETSc.

class PETScKrylovSolver
{
public:
  /// Create Krylov solver for a particular method and named
  /// preconditioner
  explicit PETScKrylovSolver(MPI_Comm comm);

  /// Create solver wrapper of a PETSc KSP object
  /// @param[in] ksp The PETSc KSP object. It should already have been created
  /// @param[in] inc_ref_count Increment the reference count on `ksp` if true
  PETScKrylovSolver(KSP ksp, bool inc_ref_count);

  // Copy constructor (deleted)
  PETScKrylovSolver(const PETScKrylovSolver& solver) = delete;

  /// Move constructor
  PETScKrylovSolver(PETScKrylovSolver&& solver);

  /// Destructor
  ~PETScKrylovSolver();

  // Assignment operator (deleted)
  PETScKrylovSolver& operator=(const PETScKrylovSolver&) = delete;

  /// Move assignment
  PETScKrylovSolver& operator=(PETScKrylovSolver&& solver);

  /// Set operator (Mat)
  void set_operator(const Mat A);

  /// Set operator and preconditioner matrix (Mat)
  void set_operators(const Mat A, const Mat P);

  /// Solve linear system Ax = b and return number of iterations (A^t x
  /// = b if transpose is true)
  int solve(Vec x, const Vec b, bool transpose = false) const;

  /// Sets the prefix used by PETSc when searching the PETSc options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the PETSc options
  /// database
  std::string get_options_prefix() const;

  /// Set options from PETSc options database
  void set_from_options() const;

  /// Return PETSc KSP pointer
  KSP ksp() const;

  /// Set the DM
  void set_dm(DM dm);

  /// Activate/deactivate DM
  void set_dm_active(bool val);

private:
  // PETSc solver pointer
  KSP _ksp;
};
} // namespace dolfinx::la
