// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005
// Last changed: 2006-08-15

#ifndef __PETSC_LU_SOLVER_H
#define __PETSC_LU_SOLVER_H

#ifdef HAS_PETSC

#include <petscmat.h>
#include <petscksp.h>

#include "GenericLinearSolver.h"
#include "PETScVector.h"

namespace dolfin
{

  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScManager;
  class PETScMatrix;
  class PETScKrylovMatrix;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of PETSc.
  
  class PETScLUSolver : public GenericLinearSolver
  {
  public:
    
    /// Constructor
    PETScLUSolver();

    /// Destructor
    ~PETScLUSolver();

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b);

    /// Solve linear system Ax = b
    uint solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b);

    /// Display LU solver data
    void disp() const;

  private:
    
    // Create dense copy of virtual matrix
    double copyToDense(const PETScKrylovMatrix& A);

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
