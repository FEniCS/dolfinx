// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
// 
// First added:  2006-08-16
// Last changed: 2006-08-16

#ifndef __LU_H
#define __LU_H

#include <dolfin/PETScLUSolver.h>
#include <dolfin/uBlasLUSolver.h>

namespace dolfin
{

  /// This class provides methods for solving a linear system by
  /// LU factorization.
  
  class LU
  {
  public:

#ifdef HAVE_PETSC_H
    
    /// Solve linear system Ax = b
    static void solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b);

    /// Solve linear system Ax = b
    static void solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b);

#endif
    
    /// Solve linear system Ax = b
    static void solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, const uBlasVector& b);
    
    /// Solve linear system Ax = b
    static void solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, const uBlasVector& b);

  };

}

#endif
