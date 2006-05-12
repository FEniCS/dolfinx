// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-08-31
// Last changed: 2005-05-12

#ifndef __EIGENVALUE_SOLVER_H
#define __EIGENVALUE_SOLVER_H

#ifdef HAVE_SLEPC_H

#include <slepceps.h>
#include <dolfin/Parametrized.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>

namespace dolfin
{

  /// This class computes eigenvalues of a matrix. It is 
	/// a wrapper for the eigenvalue solver SLEPc.
  
  class EigenvalueSolver: public Parametrized
  {
  public:

    /// Eigensolver methods
    enum Type
    { 
      arnoldi,          // Arnoldi
      default_solver,   // Default SLEPc solver (use when setting method from command line)
      lanczos,          // Lanczos
      lapack,           // LAPACK (all values, exact, only for small systems) 
      power,             // Power
      subspace          // Subspace
    };

    /// Create eigenvalue solver (use default solver type)
    EigenvalueSolver();

    /// Create eigenvalue solver (specify solver type)
    EigenvalueSolver(Type solver);

    /// Destructor
    ~EigenvalueSolver();

    /// Compute all eigenvalues of the matrix A
    void solve(const Matrix& A);

    /// Compute largest n eigenvalues of the matrix A
    void solve(const Matrix& A, const uint n);

    /// Get 0th eigenvalue/vector  
    void getEigenpair(real& xr, real& xc, Vector& r, Vector& c);

    /// Get eigenvalue/vector i 
    void getEigenpair(real& xr, real& xc, Vector& r, Vector& c, const int i);

    /// Get  0th eigenvalue 
    void getEigenvalue(real& xr, real& xc);

    /// Get eigenvalue i 
    void getEigenvalue(real& xr, real& xc, const int i);

  private:

    EPSType getType(const Type type) const;

    // SLEPc solver pointer
    EPS eps;

    /// SLEPc solver type
    Type type;

  };

}

#endif

#endif
