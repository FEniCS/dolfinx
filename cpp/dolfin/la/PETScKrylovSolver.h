// Copyright (C) 2004-2015 Johan Jansson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <string>

namespace dolfin
{
namespace fem
{
class PETScDMCollection;
}

namespace la
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
  explicit PETScKrylovSolver(KSP ksp, bool inc_ref_count = true);

  /// Destructor
  virtual ~PETScKrylovSolver();

  /// Set operator (Mat)
  void set_operator(const Mat A);

  /// Set operator and preconditioner matrix (Mat)
  void set_operators(const Mat A, const Mat P);

  /// Solve linear system Ax = b and return number of iterations (A^t x
  /// = b if transpose is true)
  int solve(Vec x, const Vec b, bool transpose = false);

  /// Sets the prefix used by PETSc when searching the PETSc options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the PETSc options
  /// database
  std::string get_options_prefix() const;

  /// Set options from PETSc options database
  void set_from_options() const;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return PETSc KSP pointer
  KSP ksp() const;

  /// Set the DM
  void set_dm(DM dm);

  /// Activate/deactivate DM
  void set_dm_active(bool val);

private:
  // Report the number of iterations
  void write_report(int num_iterations, KSPConvergedReason reason) const;

  // PETSc solver pointer
  KSP _ksp;
};
} // namespace la
} // namespace dolfin
