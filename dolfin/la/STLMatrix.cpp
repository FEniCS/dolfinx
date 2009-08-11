// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2009-08-11

#include <sstream>
#include <iomanip>

#include "STLFactory.h"
#include "STLMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::string STLMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint i = 0; i < dims[0]; i++)
    {
      std::stringstream line;
      line << std::setiosflags(std::ios::scientific);
      line << std::setprecision(16);

      line << "|";
      for (std::map<uint, double>::const_iterator it = A[i].begin(); it != A[i].end(); it++)
        line << " (" << i << ", " << it->first << ", " << it->second << ")";
      line << " |";

      s << line.str().c_str() << std::endl;
    }
  }
  else
  {
    s << "<STLMatrix of size " << size(0) << " x " << size(1) << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& STLMatrix::factory() const
{
  return STLFactory::instance();
}
//-----------------------------------------------------------------------------
