// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_SLEPC

#include "dolfinx/common/MPI.h"
#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <slepceps.h>
#include <string>

namespace dolfinx
{
namespace la
{
class VectorSpaceBasis;

/// This class provides an eigenvalue solver for PETSc matrices. It is a
/// wrapper for the SLEPc eigenvalue solver.

class SLEPcEigenSolver
{
public:
  /// Create eigenvalue solver
  explicit SLEPcEigenSolver(MPI_Comm comm);

  /// Create eigenvalue solver from EPS object
  explicit SLEPcEigenSolver(EPS eps, bool inc_ref_count = true);

  /// Destructor
  ~SLEPcEigenSolver();

  /// Set opeartors (B may be nullptr for regular eigenvalues
  /// problems)
  void set_operators(const Mat A, const Mat B);

  /// Compute all eigenpairs of the matrix A (solve Ax = \lambda x)
  void solve();

  /// Compute the n first eigenpairs of the matrix A (solve Ax = \lambda x)
  void solve(std::int64_t n);

  /// Get ith eigenvalue
  std::complex<PetscReal> get_eigenvalue(int i) const;

  /// Get ith eigenpair
  void get_eigenpair(PetscScalar& lr, PetscScalar& lc, Vec r, Vec c,
                     int i) const;

  /// Get the number of iterations used by the solver
  int get_iteration_number() const;

  /// Get the number of converged eigenvalues
  std::int64_t get_number_converged() const;

  /// Sets the prefix used by PETSc when searching the PETSc options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the PETSc
  /// options database
  std::string get_options_prefix() const;

  /// Set options from PETSc options database
  void set_from_options() const;

  /// Return SLEPc EPS pointer
  EPS eps() const;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

private:
  // SLEPc solver pointer
  EPS _eps;
};
} // namespace la
} // namespace dolfinx
#endif
