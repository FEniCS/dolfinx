// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GMRES_H
#define __GMRES_H

#include <dolfin/constants.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{
  
  class Matrix;
  class Vector;

  /// This is just a template. Write documentation here.
  
  class GMRES : public Preconditioner, public LinearSolver
  {
  public:

    /// Create GMRES preconditioner/solver for a given matrix
    GMRES(const Matrix& A, real tol, unsigned int maxiter);

    /// Destructor
    ~GMRES();

    /// Solve linear system Ax = b for a given right-hand side b
    void solve(Vector& x, const Vector& b);

    /// Solve linear system Ax = b for a given right-hand side b
    static void solve(const Matrix& A, Vector& x, const Vector& b,
		      real tol, unsigned int maxiter);

  private:

    // The matrix
    const Matrix& A;
    
    // Tolerance
    real tol;

    // Maximum number of iterations
    unsigned int maxiter;

  };

}

#endif
