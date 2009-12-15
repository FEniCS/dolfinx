// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-15
// Last changed: 2009-12-15

#include <sstream>
#include "LAPACKVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LAPACKVector::LAPACKVector(uint M) : M(M), values(new double[M])
{
  for (uint i = 0; i < M; i++)
    values[i] = 0.0;
}
//-----------------------------------------------------------------------------
LAPACKVector::~LAPACKVector()
{
  delete [] values;
}
//-----------------------------------------------------------------------------
std::string LAPACKVector::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint i = 0; i < M; i++)
    {
      s << (*this)[i];
      s << std::endl;
    }
  }
  else
  {
    s << "<LAPACK vector of size " << M << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
