// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-07-07
// Last changed: 2006-07-07

#include <dolfin/DenseVector.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/uBlasKrylovMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
/*
void uBlasKrylovMatrix::disp(const int precision) const
{
  // Since we don't really have the matrix, we create the matrix by
  // performing multiplication with unit vectors. Used only for debugging.
  
  uint M = size(0);
  uint N = size(1);
  DenseVector x(N), y(M);
  uBlasMatrix<ublas_sparse_matrix> A(M, N);
  
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
