// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-08-06

#include <sstream>
#include <iomanip> 

#include "STLFactory.h"
#include "STLMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void STLMatrix::disp(uint precision) const
{
  for (uint i = 0; i < dims[0]; i++)
  {
    std::stringstream line;
    line << std::setiosflags(std::ios::scientific);
    line << std::setprecision(precision);
    
    line << "|";
    for (std::map<uint, double>::const_iterator it = A[i].begin(); it != A[i].end(); it++)
      line << " (" << i << ", " << it->first << ", " << it->second << ")";
    line << " |";
    
    dolfin::cout << line.str().c_str() << dolfin::endl;
  }
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& STLMatrix::factory() const
{
  return STLFactory::instance(); 
}
//-----------------------------------------------------------------------------
