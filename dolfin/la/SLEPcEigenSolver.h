// Copyright (C) 2005-2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
// Modified by Marie Rognes, 2009.
//
// First added:  2005-08-31
// Last changed: 2010-09-30

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
  /// 1. "spectrum"
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
  ///
  /// For SLEPc versions >= 3.1 , the following values are also possible
  ///
  ///   "target magnitude"    (eigenvalues closest to target in magnitude)
  ///   "target real"         (eigenvalues closest to target in real part)
  ///   "target imaginary"    (eigenvalues closest to target in imaginary part)
  ///
  /// The default is "largest magnitude"
  ///
  /// 2. "solver"
  ///
  /// This parameter controls which algorithm is used by SLEPc.
  /// Possible values are
  ///
  ///   "power"               (power iteration)
  ///   "subspace"            (subspace iteration)
  ///   "arnoldi"             (Arnoldi)
  ///   "lanczos"             (Lanczos)
  ///   "krylov-schur"        (Krylov-Schur)
  ///   "lapack"              (LAPACK, all values, direct, small systems only)
  ///
  /// The default is "krylov-schur"
  ///
  /// 3. "tolerance"
  ///
  /// This parameter controls the tolerance used by SLEPc.
  /// Possible values are positive double numbers.
  ///
  /// The default is 1e-15;
  ///
  /// 4. "maximum_iterations"
  ///
  /// This parameter controls the maximum number of iterations used by SLEPc.
  /// Possible values are positive integers.
  ///
  /// Note that both the tolerance and the number of iterations must be
  /// specified if either one is specified.
  ///
  /// 5. "problem_type"
  ///
  /// This parameter can be used to give extra information about the
  /// type of the eigenvalue problem. Some solver types require this
  /// extra piece of information. Possible values are:
  ///
  ///   "hermitian"               (Hermitian)
  ///   "non_hermitian"           (Non-Hermitian)
  ///   "gen_hermitian"           (Generalized Hermitian)
  ///   "gen_non_hermitian"       (Generalized Non-Hermitian)
  ///
  /// 6. "spectral_transform"
  ///
  /// This parameter controls the application of a spectral transform. A
  /// spectral transform can be used to enhance the convergence of the
  /// eigensolver and in particular to only compute eigenvalues in the
  /// interior of the spectrum. Possible values are:
  ///
  ///   "shift-and-invert"      (A shift-and-invert transform)
  ///
  /// Note that if a spectral transform is given, then also a non-zero
  /// spectral shift parameter has to be provided.
  ///
  /// The default is no spectral transform.
  ///
  /// 7. "spectral_shift"
  ///
  /// This parameter controls the spectral shift used by the spectral
  /// transform and must be provided if a spectral transform is given. The
  /// possible values are real numbers.
  ///




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
    unsigned int system_size;

  };

}

#endif

#endif
