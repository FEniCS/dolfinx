// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
// Modified by Marie Rognes, 2009.
//
// First added:  2005-08-31
// Last changed: 2009-12-02

#ifndef __SLEPC_EIGEN_SOLVER_H
#define __SLEPC_EIGEN_SOLVER_H

#ifdef HAS_SLEPC

#include <slepceps.h>
#include "PETScObject.h"

namespace dolfin
{

  /// Forward declarations
  class PETScMatrix;
  class PETScVector;

  /// This class provides an eigenvalue solver for PETSc matrices.
  /// It is a wrapper for the SLEPc eigenvalue solver.
  ///
  /// The following parameters may be specified to control the solver.
  ///
  /// 1. "eigenvalue spectrum"
  ///
  /// This parameter controls which part of the spectrum to compute.
  /// Possible values are
  ///
  ///   "largest magnitude"   (eigenvalues with largest magnitude)
  ///   "smallest magnitude"  (eigenvalues with smallest magnitude)
  ///   "largest real"        (eigenvalues with largest double part)
  ///   "smallest real"       (eigenvalues with smallest double part)
  ///   "largest imaginary"   (eigenvalues with largest imaginary part)
  ///   "smallest imaginary"  (eigenvalues with smallest imaginary part)
  ///   "default spectrum"    (default spectrum)
  ///
  /// 2. "eigenvalue solver"
  ///
  /// This parameter controls which algorithm is used by SLEPc.
  /// Possible values are
  ///
  ///   "power"               (power iteration)
  ///   "subspace"            (subspace iteration)
  ///   "arnoldi"             (Arnoldi)
  ///   "lanczos"             (Lanczos)
  ///   "krylov-schur"        (Krylov-Schur)
  ///   "lapack"              (LAPACK, all values, direct, only for small systems)
  ///   "default"             (default algorithm)
  ///
  /// 3. "eigenvalue tolerance"
  ///
  /// This parameter controls the tolerance used by SLEPc.
  /// Possible values are positive double numbers.
  ///
  /// 4. "eigenvalue iterations"
  ///
  /// This parameter controls the maximum number of iterations used by SLEPc.
  /// Possible values are positive integers.
  ///
  /// Note that both the tolerance and the number of iterations must be
  /// specified if either one is specified.

  class SLEPcEigenSolver : public Variable, public PETScObject
  {
  public:

    /// Create eigenvalue solver
    SLEPcEigenSolver();

    /// Destructor
    ~SLEPcEigenSolver();

    /// Compute all eigenpairs of the matrix A (solve Ax = \lambda x)
    void solve(const PETScMatrix& A);

    /// Compute the n first eigenpairs of the matrix A (solve Ax = \lambda x)
    void solve(const PETScMatrix& A, uint n);

    /// Compute all eigenpairs of the generalised problem Ax = \lambda Bx
    void solve(const PETScMatrix& A, const PETScMatrix& B);

    /// Compute the n first eigenpairs of the generalised problem Ax = \lambda Bx
    void solve(const PETScMatrix& A, const PETScMatrix& B, uint n);

    /// Get the first eigenvalue
    void get_eigenvalue(double& lr, double& lc);

    /// Get the first eigenpair
    void get_eigenpair(double& lr, double& lc, PETScVector& r, PETScVector& c);

    /// Get eigenvalue i
    void get_eigenvalue(double& lr, double& lc, uint i);

    /// Get eigenpair i
    void get_eigenpair(double& lr, double& lc, PETScVector& r, PETScVector& c, uint i);

    // Get the number of iterations used by the solver
    int get_iteration_number();

    // Get the number of converged eigenvalues
    int get_number_converged();

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("slepc_eigenvalue_solver");

      p.add("problem_type",       "default");
      p.add("spectrum",           "largest magnitude");
      p.add("solver",             "krylov-schur");
      p.add("tolerance",          1e-15);
      p.add("maximum_iterations", 10000);
      p.add("spectral_transform", "default");
      p.add("spectral_shift",     0.0);

      return p;
    }

  private:

    /// Compute eigenpairs
    void solve(const PETScMatrix* A, const PETScMatrix* B, uint n);

    /// Callback for changes in parameter values
    void read_parameters();

    // Set problem type (used for SLEPc internals)
    void set_problem_type(std::string type);

    // Set spectral transform
    void set_spectral_transform(std::string transform, double shift);

    // Set spectrum
    void set_spectrum(std::string solver);

    // Set solver
    void set_solver(std::string spectrum);

    // Set tolerance
    void set_tolerance(double tolerance, uint maxiter);

    // SLEPc solver pointer
    EPS eps;

    // System size
    uint system_size;

  };

}

#endif

#endif
