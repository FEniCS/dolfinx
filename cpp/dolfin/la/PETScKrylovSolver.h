// Copyright (C) 2004-2015 Johan Jansson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include "PETScObject.h"
#include <dolfin/common/types.h>
#include <map>
#include <memory>
#include <petscksp.h>
#include <string>

namespace dolfin
{
namespace fem
{
class PETScDMCollection;
}

namespace la
{
class PETScBaseMatrix;
class PETScMatrix;
class PETScVector;
class VectorSpaceBasis;

/// This class implements Krylov methods for linear systems of the
/// form Ax = b. It is a wrapper for the Krylov solvers of PETSc.

class PETScKrylovSolver : public PETScObject
{
public:
  /// Norm types used in convergence testing. Not all solvers types
  /// support all norm types (see
  /// http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetNormType.html).
  /// Note
  /// that 'default' is a reserved keyword, so we use 'default_norm'
  enum class norm_type
  {
    none,
    default_norm,
    preconditioned,
    unpreconditioned,
    natural
  };

  /// Create Krylov solver for a particular method and named
  /// preconditioner
  PETScKrylovSolver(MPI_Comm comm, std::string method = "default",
                    std::string preconditioner = "default");

  /// Create solver wrapper of a PETSc KSP object
  explicit PETScKrylovSolver(KSP ksp);

  /// Destructor
  virtual ~PETScKrylovSolver();

  /// Set operator (PETScMatrix). This is memory-safe as PETSc will
  /// increase the reference count to the underlying PETSc object.
  void set_operator(const la::PETScBaseMatrix& A);

  /// Set operator and preconditioner matrix (PETScMatrix). This is
  /// memory-safe as PETSc will increase the reference count to the
  /// underlying PETSc objects.
  void set_operators(const la::PETScBaseMatrix& A,
                     const la::PETScBaseMatrix& P);

  /// Solve linear system Ax = b and return number of iterations
  /// (A^t x = b if transpose is true)
  std::size_t solve(PETScVector& x, const PETScVector& b,
                    bool transpose = false);

  /// Use nonzero initial guess for solution function
  /// (nonzero_guess=true, the solution vector x will not be zeroed
  /// before the solver starts)
  void set_nonzero_guess(bool nonzero_guess);

  /// Reuse preconditioner if true, even if matrix operator changes
  /// (by default preconditioner will be re-built if the matrix
  /// changes)
  void set_reuse_preconditioner(bool reuse_pc);

  /// Set tolerances (relative residual, alsolute residial, maximum
  /// number of iterations)
  void set_tolerances(double relative, double absolute, double diverged,
                      int max_iter);

  /// Set norm type used in convergence testing - not all solvers
  /// types support all norm types
  void set_norm_type(norm_type type);

  /// Get norm type used in convergence testing
  norm_type get_norm_type() const;

  /// Monitor residual at each iteration
  void monitor(bool monitor_convergence);

  /// Sets the prefix used by PETSc when searching the PETSc options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the PETSc
  /// options database
  std::string get_options_prefix() const;

  /// Set options from PETSc options database
  void set_from_options() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return PETSc KSP pointer
  KSP ksp() const;

  /// Return a list of names solver methods. For access to all
  /// methods, use PETScOptions.
  static std::map<std::string, std::string> methods();

  /// Return a list of named preconditioner shortcuts. For access to
  /// all methods, use PETScOptions.
  static std::map<std::string, std::string> preconditioners();

  /// Set the DM
  void set_dm(DM dm);

  /// Activate/deactivate DM
  void set_dm_active(bool val);

  friend class PETScSNESSolver;
  friend class PETScTAOSolver;

private:
  // Temporary work-arounds
  static std::map<std::string, const KSPType> petsc_methods();
  static std::map<std::string, const PCType> petsc_pc_methods();

  // Return norm_type enum for norm string
  static PETScKrylovSolver::norm_type get_norm_type(std::string norm);

  // Solve linear system Ax = b and return number of iterations
  std::size_t _solve(const la::PETScBaseMatrix& A, PETScVector& x,
                     const PETScVector& b);

  // Report the number of iterations
  void write_report(int num_iterations, KSPConvergedReason reason);

  void check_dimensions(const la::PETScBaseMatrix& A, const PETScVector& x,
                        const PETScVector& b) const;

  // PETSc solver pointer
  KSP _ksp;

  bool preconditioner_set;
};
}
}
#endif
