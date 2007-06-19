// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-30
// Last changed: 2007-04-30

#ifndef __SOLVE_H
#define __SOLVE_H

#include <dolfin/constants.h>

namespace dolfin
{  

  /// Solve linear system Ax = b using a direct method (LU factorization).
  /// Note that iterative methods (preconditioned Krylov methods including
  /// GMRES) are also available through the KrylovSolver interface.

#ifdef HAVE_PETSC_H
  
  /// Solve linear system Ax = b
  void solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b);
  
  /// Solve linear system Ax = b
  void solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b);
  
#endif
  
  /// Solve linear system Ax = b
  void solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, const uBlasVector& b);
  
  /// Solve linear system Ax = b
  void solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, const uBlasVector& b);

}

#endif
