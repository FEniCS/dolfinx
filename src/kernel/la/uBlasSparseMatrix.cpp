// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-03
// Last changed: 2006-07-05


#include <dolfin/uBlasSparseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const uBlasSparseMatrix& A)
{
  // Check if matrix has been defined
  if ( A.size(0) == 0 || A.size(1) == 0 )
  {
    stream << "[ uBlasSparseMatrix matrix (empty) ]";
    return stream;
  }

  uint M = A.size(0);
  uint N = A.size(1);
  stream << "[ uBlasSparseMatrix matrix of size " << M << " x " << N << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
