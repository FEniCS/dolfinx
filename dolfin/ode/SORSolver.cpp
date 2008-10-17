// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-11
// Last changed: 2008-10-11

#include "SORSolver.h"
#include <dolfin/la/uBLASDenseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void SORSolver::SOR(uint n, 
		    const real* A, 
		    real* x, 
		    const real* b, 
		    const real& epsilon) 
{
  real prev[n];
    
  uint iterations = 0;
  real diff = 1000.0; //some 
  
  uint count = 0;

  while ( diff > epsilon ) 
  {
    ++count;

    if ( iterations > SOR_MAX_ITERATIONS ) 
    {
      error("SOR: System seems not to converge");
    }
    real_set(n, prev, x);
    
    _SOR_iteration(n, A, b, x, prev);

    //check precision
    real_sub(n, prev, x);
    diff = real_max_abs(n, prev);
    ++iterations;

  }
  //printf ("Iterations: %d\n", count);
}
//-----------------------------------------------------------------------------
void SORSolver::_SOR_iteration(uint n, 
			  const real* A, 
			  const real* b, 
			  real* x_new, 
			  const real* x_prev)
{
    
  for (uint i = 0; i < n; ++i) 
  {
    x_new[i] = b[i]/A[i*n+i];

    // j < i
    for (uint j = 0; j < i; ++j) 
    { x_new[i] -= (A[i*n+j]/A[i*n+i])*x_new[j]; }
    
    // j > i
    for (uint j = i+1; j < n; ++j) 
    { x_new[i] -= (A[i*n+j]/A[i*n+i])*x_prev[j]; }
  }
}
//-----------------------------------------------------------------------------
void SORSolver::precondition(uint n, const uBLASDenseMatrix& A_inv, 
			     real* A, real* b,
			     real* Ainv_A, real* Ainv_b) {
  
  //Compute A*A_inv and A_inv*b
  //with extended precision
  for (uint i=0; i<n; ++i) {
    Ainv_b[i] = 0.0;
    for (uint j=0; j<n; ++j) {
      Ainv_b[i] += A_inv(i,j)*b[j];
      
      Ainv_A[i*n+j] = 0.0;
      for (uint k=0; k<n; ++k) {
	Ainv_A[i*n+j] += A[i*n+k]*A_inv(k,j);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SORSolver::printMatrix(const uint n, const real* A)
{
  for (uint i = 0; i<n; ++i)
  {
    for (uint j = 0; j<n; ++j) 
      { printf("(%d, %d, %f) ", i, j, to_double(A[i*n+j])); }
    printf("\n");
  }
}
//-----------------------------------------------------------------------------
void SORSolver::printVector (const uint n, const real* x)
{
  for (uint i = 0; i<n; ++i) printf("%f\n", to_double(x[i]));
  printf("\n");
}
//-----------------------------------------------------------------------------
