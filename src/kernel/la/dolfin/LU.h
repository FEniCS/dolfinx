// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2006-05-15

#ifndef __LU_H
#define __LU_H

#ifdef HAVE_PETSC_H

#include <dolfin/PETScManager.h>
#include <dolfin/LinearSolver.h>
#include <dolfin/Parametrized.h>

namespace dolfin
{
  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of PETSc.
  
  class LU : public LinearSolver, public Parametrized
  {
  public:
    
    /// Constructor
    LU();

    /// Destructor
    ~LU();

    /// Solve linear system Ax = b
    uint solve(const PETScSparseMatrix& A, PETScVector& x, const PETScVector& b);

    /// Solve linear system Ax = b
    uint solve(const VirtualMatrix& A, PETScVector& x, const PETScVector& b);

    /// Display LU solver data
    void disp() const;

  private:
    
    // Create dense copy of virtual matrix
    real copyToDense(const VirtualMatrix& A);

    KSP ksp;

    Mat B;
    int* idxm;
    int* idxn;

    PETScVector e;
    PETScVector y;

  };

}

#endif

#endif
