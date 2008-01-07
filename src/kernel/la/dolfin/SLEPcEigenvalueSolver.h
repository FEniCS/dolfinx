// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-08-31
// Last changed: 2006-08-21

#ifndef __SLEPC_EIGENVALUE_SOLVER_H
#define __SLEPC_EIGENVALUE_SOLVER_H

#ifdef HAVE_SLEPC_H

#include <slepceps.h>
#include <dolfin/Parametrized.h>

namespace dolfin
{

  /// Forward declarations
  class PETScMatrix;
  class PETScVector;

  /// This class computes eigenvalues of a matrix. It is 
	/// a wrapper for the eigenvalue solver SLEPc.
  
  class SLEPcEigenvalueSolver: public Parametrized
  {
  public:

    /// Eigensolver methods
    enum Type
    { 
      arnoldi,          // Arnoldi
      default_solver,   // Default SLEPc solver (use when setting method from command line)
      lanczos,          // Lanczos
      lapack,           // LAPACK (all values, exact, only for small systems) 
      power,            // Power
      subspace          // Subspace
    };

    /// Create eigenvalue solver (use default solver type)
    SLEPcEigenvalueSolver();

    /// Create eigenvalue solver (specify solver type)
    SLEPcEigenvalueSolver(Type solver);

    /// Destructor
    ~SLEPcEigenvalueSolver();

    /// Compute all eigenpairs of the matrix A (solve Ax = \lambda x)
    void solve(const PETScMatrix& A);

    /// Compute largest n eigenpairs of the matrix A (solve Ax = \lambda x)
    void solve(const PETScMatrix& A, uint n);

    /// Compute all eigenpairs of the generalised problem Ax = \lambda Bx
    void solve(const PETScMatrix& A, const PETScMatrix& B);

    /// Compute largest n eigenpairs of the generalised problem Ax = \lambda Bx
    void solve(const PETScMatrix& A, const PETScMatrix& B, uint n);

    /// Get the 0th eigenvalue 
    void getEigenvalue(real& xr, real& xc);

    /// Get 0th eigenpair  
    void getEigenpair(real& xr, real& xc, PETScVector& r, PETScVector& c);

    /// Get eigenvalue i 
    void getEigenvalue(real& xr, real& xc, const int i);

    /// Get eigenpair i 
    void getEigenpair(real& xr, real& xc, PETScVector& r, PETScVector& c, const int i);

  private:

    /// Compute eigenvalues
    void solve(const PETScMatrix& A, const PETScMatrix* B, uint n);

    EPSType getType(const Type type) const;

    // SLEPc solver pointer
    EPS eps;

    /// SLEPc solver type
    Type type;

  };

}

#endif

#endif
