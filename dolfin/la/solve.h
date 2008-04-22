// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-04-30
// Last changed: 2008-04-11

#ifndef __SOLVE_H
#define __SOLVE_H

#include <dolfin/common/types.h>

namespace dolfin
{  

  class GenericMatrix;
  class GenericVector;

  /// Solve linear system Ax = b using a direct method (LU factorization).
  /// Note that iterative methods (preconditioned Krylov methods including
  /// GMRES) are also available through the KrylovSolver interface.

  /// Solve linear system Ax = b
  void solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

  /// Compute residual ||Ax - b||
  real residual(const GenericMatrix& A, const GenericVector& x, const GenericVector& b);
  
  /// Solve linear system Ax = b
  //void solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b);
  
}

#endif
