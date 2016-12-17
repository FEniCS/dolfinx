// Copyright (C) 2005-2014 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
// Modified by Marie Rognes, 2009.

#ifndef __SLEPC_EIGEN_SOLVER_H
#define __SLEPC_EIGEN_SOLVER_H

#ifdef HAS_SLEPC

#include <string>
#include <memory>
#include <slepceps.h>
#include "dolfin/common/types.h"
#include "dolfin/common/MPI.h"
#include "PETScObject.h"

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  class PETScMatrix;
  class PETScVector;
  class VectorSpaceBasis;

  /// This class provides an eigenvalue solver for PETSc matrices. It
  /// is a wrapper for the SLEPc eigenvalue solver.
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
  ///   "target magnitude"    (eigenvalues closest to target in magnitude)
  ///   "target real"         (eigenvalues closest to target in real part)
  ///   "target imaginary"    (eigenvalues closest to target in imaginary part)
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
  ///   "arpack"              (ARPACK)
  ///
  /// 3. "tolerance"
  ///
  /// This parameter controls the tolerance used by SLEPc.  Possible
  /// values are positive double numbers.
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
  ///   "pos_gen_non_hermitian"   (Generalized Non-Hermitian with positive semidefinite B)
  ///
  /// 6. "spectral_transform"
  ///
  /// This parameter controls the application of a spectral
  /// transform. A spectral transform can be used to enhance the
  /// convergence of the eigensolver and in particular to only compute
  /// eigenvalues in the interior of the spectrum. Possible values
  /// are:
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
  /// transform and must be provided if a spectral transform is
  /// given. The possible values are real numbers.

  class SLEPcEigenSolver : public Variable, public PETScObject
  {
  public:

    /// Create eigenvalue solver
    explicit SLEPcEigenSolver(MPI_Comm comm);

    /// Create eigenvalue solver from EPS object
    explicit SLEPcEigenSolver(EPS eps);

    /// Create eigenvalue solver for Ax = \lambda
    explicit SLEPcEigenSolver(std::shared_ptr<const PETScMatrix> A);

    /// Create eigenvalue solver for Ax = \lambda x
    SLEPcEigenSolver(MPI_Comm comm, std::shared_ptr<const PETScMatrix> A);

    /// Create eigenvalue solver for Ax = \lambda x on MPI_COMM_WORLD
    SLEPcEigenSolver(std::shared_ptr<const PETScMatrix> A,
                     std::shared_ptr<const PETScMatrix> B);

    /// Create eigenvalue solver for Ax = \lambda x
    SLEPcEigenSolver(MPI_Comm comm, std::shared_ptr<const PETScMatrix> A,
                     std::shared_ptr<const PETScMatrix> B);

    /// Destructor
    ~SLEPcEigenSolver();

    /// Set opeartors (B may be nullptr for regular eigenvalues
    /// problems)
    void set_operators(std::shared_ptr<const PETScMatrix> A,
                       std::shared_ptr<const PETScMatrix> B);

    /// Compute all eigenpairs of the matrix A (solve Ax = \lambda x)
    void solve();

    /// Compute the n first eigenpairs of the matrix A (solve Ax = \lambda x)
    void solve(std::size_t n);

    /// Get the first eigenvalue
    void get_eigenvalue(double& lr, double& lc) const;

    /// Get the first eigenpair
    void get_eigenpair(double& lr, double& lc,
                       GenericVector& r, GenericVector& c) const;

    /// Get the first eigenpair
    void get_eigenpair(double& lr, double& lc,
                       PETScVector& r, PETScVector& c) const;

    /// Get eigenvalue i
    void get_eigenvalue(double& lr, double& lc, std::size_t i) const;

    /// Get eigenpair i
    void get_eigenpair(double& lr, double& lc,
                       GenericVector& r, GenericVector& c, std::size_t i) const;

    /// Get eigenpair i
    void get_eigenpair(double& lr, double& lc,
                       PETScVector& r, PETScVector& c, std::size_t i) const;

    /// Get the number of iterations used by the solver
    std::size_t get_iteration_number() const;

    /// Get the number of converged eigenvalues
    std::size_t get_number_converged() const;

    /// Set deflation space
    void set_deflation_space(const PETScVector& deflation_space);

    /// Set deflation space. The VectorSpaceBasis used for deflation need
    /// not be an orthonormal set.
    void set_deflation_space(const VectorSpaceBasis& deflation_space);

    /// Sets the prefix used by PETSc when searching the PETSc options
    /// database
    void set_options_prefix(std::string options_prefix);

    /// Returns the prefix used by PETSc when searching the PETSc
    /// options database
    std::string get_options_prefix() const;

    /// Return SLEPc EPS pointer
    EPS eps() const;

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("slepc_eigenvalue_solver");
      p.add<std::string>("problem_type");
      p.add<std::string>("spectrum");
      p.add<std::string>("solver");
      p.add<double>("tolerance");
      p.add<int>("maximum_iterations");
      p.add<std::string>("spectral_transform");
      p.add<double>("spectral_shift");
      p.add<bool>("verbose");

      return p;
    }

  private:

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
    void set_tolerance(double tolerance, int maxiter);

    // Operators (A x = \lambda x or Ax = \lambda B x)
    std::shared_ptr<const PETScMatrix> _matA, _matB;

    // SLEPc solver pointer
    EPS _eps;

  };

}

#endif

#endif
