// Copyright (C) 2011 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-16
// Last changed: 2011-02-16

#ifndef __HIGH_PRECISION_H
#define __HIGH_PRECISION_H


#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include <dolfin/la/uBLASDenseMatrix.h>

#define MAX_ITERATIONS 1000

namespace dolfin
{

  // Matrix multiplication res = A*B
  void real_mat_prod(uint n, real* res, const real* A, const real* B);

  // Matrix multiplication A = A * B
  void real_mat_prod_inplace(uint n, real* A, const real* B);

  // Matrix vector product y = Ax
  void real_mat_vector_prod(uint n, real* y, const real* A, const real* x);

  // Matrix power A = B^q
  void real_mat_pow(uint n, real* A, const real* B, uint q);

  
  // Functions for solving linear systems with high precision


  // Solve AX = B by first inverting A in double precision, then 
  // doing Gauss-Seidel iteration with A inverted as preconditioner.
  // Will replace the initial guess in x
  void real_solve_mat_with_preconditioning(uint n,
                                    const real* A,
                                    real* x,
                                    const real* B,
                                    const real& tol);


  // Solves Ax = b by Gauss-Seidel iterations
  void real_solve(uint n, const real* A, real* x, const real* b, const real& tol);

  // Solve Ax = b with preconditioner
  void real_solve_precond(uint n, 
                     const real* A, 
                     real* x, 
                     const real* b, 
                     const uBLASDenseMatrix& precond,
                     const real& tol );
}
#endif
