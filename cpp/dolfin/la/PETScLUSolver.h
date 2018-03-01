// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include "PETScKrylovSolver.h"
#include <dolfin/common/MPI.h>
#include <map>
#include <memory>
#include <petscmat.h>
#include <petscpc.h>
#include <string>

// Temporary fix for PETSc master
#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 8                       \
    && PETSC_VERSION_SUBMINOR >= 3 && PETSC_VERSION_RELEASE == 0
#define MatSolverPackage MatSolverType
#define PCFactorSetMatSolverPackage PCFactorSetMatSolverType
#define PCFactorGetMatSolverPackage PCFactorGetMatSolverType
#endif

namespace dolfin
{
namespace la
{
class PETScLinearOperator;
class PETScMatrix;
class PETScVector;

/// This class implements the direct solution (LU factorization) for
/// linear systems of the form Ax = b. It is a wrapper for the LU
/// solver of PETSc.

class PETScLUSolver
{
public:
  /// Constructor
  PETScLUSolver(MPI_Comm comm, std::string method = "default");

  /// Constructor
  PETScLUSolver(std::string method = "default");

  /// Constructor
  PETScLUSolver(MPI_Comm comm, std::shared_ptr<const PETScMatrix> A,
                std::string method = "default");

  /// Destructor
  ~PETScLUSolver();

  /// Set operator (matrix)
  void set_operator(const PETScMatrix& A);

  /// Solve linear system Ax = b
  std::size_t solve(PETScVector& x, const PETScVector& b);

  /// Solve linear system Ax = b (A^t x = b if transpose is true)
  std::size_t solve(PETScVector& x, const PETScVector& b, bool transpose);

  /// Sets the prefix used by PETSc when searching the options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Set options from the PETSc options database
  void set_from_options() const;

  /// Returns the MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Return PETSc KSP pointer
  KSP ksp() const;

  /// Return a list of available solver methods
  static std::map<std::string, std::string> methods();

  /// Return parameter type: "krylov_solver" or "lu_solver"
  std::string parameter_type() const { return "lu_solver"; }

  // FIXME: These should not be friend classes
  friend class PETScSNESSolver;
  friend class PETScTAOSolver;

private:
  // FIXME: Remove
  // Available LU solvers
  static std::map<std::string, const MatSolverPackage> lumethods;

  // Select LU solver type
  static const MatSolverPackage select_solver(MPI_Comm comm,
                                              std::string method);

  PETScKrylovSolver _solver;
};
}
}
#endif
