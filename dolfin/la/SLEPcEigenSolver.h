// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2005-08-31
// Last changed: 2008-08-27

#ifndef __SLEPC_EIGEN_SOLVER_H
#define __SLEPC_EIGEN_SOLVER_H

#ifdef HAS_SLEPC

#include <slepceps.h>
#include <dolfin/parameter/Parametrized.h>
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
  ///   "largest real"        (eigenvalues with largest real part)
  ///   "smallest real"       (eigenvalues with smallest real part)
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
  /// Possible values are positive real numbers.
  ///
  /// 4. "eigenvalue iterations"
  ///
  /// This parameter controls the maximum number of iterations used by SLEPc.
  /// Possible values are positive integers.
  ///
  /// Note that both the tolerance and the number of iterations must be
  /// specified if either one is specified.
  
  class SLEPcEigenSolver : public Parametrized, public PETScObject
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
    void getEigenvalue(real& lr, real& lc);

    /// Get the first eigenpair
    void getEigenpair(real& lr, real& lc, PETScVector& r, PETScVector& c);

    /// Get eigenvalue i
    void getEigenvalue(real& lr, real& lc, uint i);

    /// Get eigenpair i
    void getEigenpair(real& lr, real& lc, PETScVector& r, PETScVector& c, uint i);

  private:

    /// Compute eigenpairs
    void solve(const PETScMatrix* A, const PETScMatrix* B, uint n);

    /// Callback for changes in parameter values
    void readParameters();
    
    // Set spectrum
    void setSpectrum(std::string solver);

    // Set solver
    void setSolver(std::string spectrum);

    // Set tolerance
    void setTolerance(double tolerance, uint maxiter);

    // SLEPc solver pointer
    EPS eps;

  };

}

#endif

#endif
