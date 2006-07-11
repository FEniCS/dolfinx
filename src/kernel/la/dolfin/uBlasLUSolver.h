// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed: 2006-07-10

#ifndef __UBLAS_LU_SOLVER_H
#define __UBLAS_LU_SOLVER_H

#include <dolfin/Parametrized.h>
#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/LinearSolver.h>


namespace dolfin
{
  class DenseVector;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b using uBlas data types.
  
  class uBlasLUSolver : public LinearSolver, public Parametrized
  {
  public:
    
    /// Constructor
    uBlasLUSolver();

    /// Destructor
    ~uBlasLUSolver();

    /// Solve linear system Ax = b (A is dense)
    static uint solve(const uBlasDenseMatrix& A, DenseVector& x, const DenseVector& b);

    /// Solve linear system Ax = b in place (A is dense)
    static uint solveInPlace(uBlasDenseMatrix& A, DenseVector& x, const DenseVector& b);

    /// Compute the inverse of A (A is dense)
    static void invert(uBlasDenseMatrix& A);

    /// Solve linear system Ax = b (A is sparse)
    /// UMFPACK is used if it has been configured. Otherwise a Krylov is used.
    uint solve(const uBlasSparseMatrix& A, DenseVector& x, const DenseVector& b) const;

  private:
    
  };

}

#endif
