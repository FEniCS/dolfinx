// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GAUSS_SEIDEL_H
#define __GAUSS_SEIDEL_H

#include <dolfin/constants.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{
  
  class Matrix;
  class Vector;

  /// This class implements the standard Gauss-Seidel method for a
  /// linear system. It implements the interface specified by the
  /// class Preconditioner. It also provides a static method for the
  /// solution of a given linear system. It may thus be used both as
  /// a preconditioner and as a linear solver.
    
  class GaussSeidel : public Preconditioner, public LinearSolver
  {
  public:

    /// Create Gauss-Seidel preconditioner/solver for a given matrix
    GaussSeidel(const Matrix& A, real tol, unsigned int maxiter);

    /// Create Gauss-Seidel preconditioner/solver for a given matrix
    GaussSeidel(const Matrix& A, unsigned int maxiter);

    /// Destructor
    ~GaussSeidel();

    /// Solve linear system Ax = b for a given right-hand side b
    void solve(Vector& x, const Vector& b);

    /// Solve linear system Ax = b for a given right-hand side b
    static void solve(const Matrix& A, Vector& x, const Vector& b,
		      real tol, unsigned int maxiter);

  private:
    
    // Perform one Gauss-Seidel iteration
    void iteration(const Matrix& A, Vector& x, const Vector& b);
    
    // The matrix
    const Matrix& A;

    // Tolerance
    real tol;

    // Maximum number of iterations
    unsigned int maxiter;

  };

}

#endif
