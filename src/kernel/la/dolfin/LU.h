// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LU_H
#define __LU_H

#include <petscksp.h>
#include <petscmat.h>

#include <dolfin/LinearSolver.h>

namespace dolfin
{

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of PETSc.
  
  class LU : public LinearSolver
  {
  public:
    
    /// Constructor
    LU();

    /// Destructor
    ~LU();

    /// Solve linear system Ax = b
    void solve(const Matrix& A, Vector& x, const Vector& b);

    /// Solve linear system Ax = b (matrix-free version)
    void solve(const VirtualMatrix& A, Vector& x, const Vector& b);

    /// Display LU solver data
    void disp() const;

  private:
    
    // Create dense copy of virtual matrix
    real copyToDense(const VirtualMatrix& A);

    KSP ksp;

    Mat B;
    int* idxm;
    int* idxn;

    Vector e;
    Vector y;

  };

}

#endif
