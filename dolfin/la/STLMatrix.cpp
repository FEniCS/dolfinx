// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2010-11-08

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
      const std::vector<uint>& rcols = this->cols[i];
      const std::vector<double>& rvals = this->vals[i];

      std::stringstream line;
      line << std::setiosflags(std::ios::scientific);
      line << std::setprecision(16);

      line << "|";
      for (uint k = 0; k < rcols.size(); k++)
        line << " (" << i << ", " << rcols[k] << ", " << rvals[k] << ")";
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
