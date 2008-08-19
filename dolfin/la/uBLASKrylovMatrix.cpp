// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-07-07
// Last changed: 2006-07-07

#include "uBLASVector.h"
#include "uBLASSparseMatrix.h"
#include "uBLASKrylovMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void uBLASKrylovMatrix::solve(uBLASVector& x, const uBLASVector& b)
{
  // The linear system is solved by computing a dense copy of the matrix,
  // obtained through multiplication with unit vectors.

  // Check dimensions
  const uint M  = size(0);
  const uint N  = size(1);
  dolfin_assert(M == N);
  dolfin_assert(M == b.size());

  // Initialize temporary data if not already done
  if ( !AA )
  {
    AA = new uBLASMatrix<ublas_dense_matrix>(M, N);
    ej = new uBLASVector(N);
    Aj = new uBLASVector(M);
  }
  else
  {
    AA->init(M, N);
    ej->init(N);
    Aj->init(N);
  }

  // Get underlying uBLAS vectors
  ublas_vector& _ej = ej->vec(); 
  ublas_vector& _Aj = Aj->vec(); 
  ublas_dense_matrix& _AA = AA->mat(); 

  // Reset unit vector
  _ej *= 0.0;

  // Compute columns of matrix
  for (uint j = 0; j < N; j++)
  {
    (_ej)(j) = 1.0;

    // Compute product Aj = Aej
    mult(*ej, *Aj);
    
    // Set column of A
    column(_AA, j) = _Aj;
    
    (_ej)(j) = 0.0;
  }

  // Solve linear system
  warning("UmfpackLUSolver no longer solves dense matrices. This function will be removed and probably added to uBLASKrylovSolver.");
  warning("The uBLASKrylovSolver LU solver has been modified and has not yet been well tested. Please verify your results.");
 (*AA).solve(x, b);
}
//-----------------------------------------------------------------------------
/*
void uBLASKrylovMatrix::disp(const int precision) const
{
  // Since we don't really have the matrix, we create the matrix by
  // performing multiplication with unit vectors. Used only for debugging.
  
  uint M = size(0);
  uint N = size(1);
  uBLASVector x(N), y(M);
  uBLASMatrix<ublas_sparse_matrix> A(M, N);
  
  x = 0.0;
  for (unsigned int j = 0; j < N; j++)
  {
    x(j) = 1.0;
    mult(x, y);
    for (unsigned int i = 0; i < M; i++)
    {
      const real value = y(i);
      if ( fabs(value) > DOLFIN_EPS )
        A(i, j) = value;
    }
    x(j) = 0.0;
  }

  A.disp(precision);
}
//-----------------------------------------------------------------------------
*/
