// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LU_H
#define __LU_H

#include <petscksp.h>
#include <petscmat.h>

#include <dolfin/NewLinearSolver.h>

namespace dolfin
{

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of PETSc.
  
  class LU : public NewLinearSolver
  {
  public:
    
    /// Constructor
    LU();

    /// Destructor
    ~LU();

    /// Solve linear system Ax = b
    void solve(const NewMatrix& A, NewVector& x, const NewVector& b);

    /// Solve linear system Ax = b (matrix-free version)
    void solve(const VirtualMatrix& A, NewVector& x, const NewVector& b);

    /// Display LU solver data
    void disp() const;

  private:
    
    // Create dense copy of virtual matrix
    void copyToDense(const VirtualMatrix& A);

    KSP ksp;

    Mat B;
    int* idxm;
    int* idxn;

    NewVector e;
    NewVector y;

  };

}

#endif
