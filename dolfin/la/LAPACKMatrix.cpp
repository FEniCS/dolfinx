// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-15
// Last changed: 2009-12-15

#include <sstream>
#include "LAPACKMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LAPACKMatrix::LAPACKMatrix(uint M, uint N)
  : M(M), N(N), values(new double[N*M])
{
  for (uint i = 0; i < M*N; i++)
    values[i] = 0.0;
}
//-----------------------------------------------------------------------------
LAPACKMatrix::~LAPACKMatrix()
{
  delete [] values;
}
//-----------------------------------------------------------------------------
std::string LAPACKMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint i = 0; i < M; i++)
    {
      for (uint j = 0; j < N; j++)
      {
        s << (*this)(i, j);
        if (j < N - 1) s << " ";
      }
      s << std::endl;
    }
  }
  else
  {
    s << "<LAPACK matrix of size " << M << " x " << N << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
