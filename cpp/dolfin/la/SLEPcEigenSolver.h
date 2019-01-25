// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_SLEPC

#include "dolfin/common/MPI.h"
#include "dolfin/common/types.h"
#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <slepceps.h>
#include <string>

namespace dolfin
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
  explicit SLEPcEigenSolver(EPS eps);

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
  std::complex<PetscReal> get_eigenvalue(std::size_t i) const;

  /// Get ith eigenpair
  void get_eigenpair(PetscScalar& lr, PetscScalar& lc, Vec r, Vec c,
                     std::size_t i) const;

  /// Get the number of iterations used by the solver
  std::size_t get_iteration_number() const;

  /// Get the number of converged eigenvalues
  std::size_t get_number_converged() const;

  /// Set deflation space. The VectorSpaceBasis does not need to be
  /// orthonormal.
  void set_deflation_space(const la::VectorSpaceBasis& deflation_space);

  /// Set inital space. The VectorSpaceBasis does not need to be
  /// orthonormal.
  void set_initial_space(const la::VectorSpaceBasis& initial_space);

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
  // Set problem type (used for SLEPc internals)
  void set_problem_type(std::string type);

  // Set spectral transform
  void set_spectral_transform(std::string transform, double shift);

  // Set solver
  void set_solver(std::string spectrum);

  // Set tolerance
  void set_tolerance(double tolerance, int maxiter);

  // SLEPc solver pointer
  EPS _eps;
};
} // namespace la
} // namespace dolfin
#endif
