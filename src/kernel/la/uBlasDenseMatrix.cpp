// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-07-04
// Last changed: 

#include <dolfin/uBlasDenseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasDenseMatrix::uBlasDenseMatrix() : uBlasMatrix<ublas_dense_matrix>()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasDenseMatrix::uBlasDenseMatrix(const uint M, const uint N) 
                  : uBlasMatrix<ublas_dense_matrix>(M, N)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const uBlasDenseMatrix& A)
{
  const uint M = A.size(0);
  const uint N = A.size(1);

  // Check if matrix has been defined
  if ( M == 0 || N == 0 )
  {
    stream << "[ uBlasDenseMatrix matrix (empty) ]";
    return stream;
  }

  stream << "[ uBlasDenseMatrix matrix of size " << M << " x " << N << " ]";
  return stream;
}
//-----------------------------------------------------------------------------
