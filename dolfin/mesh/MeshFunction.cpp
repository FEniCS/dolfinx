// Copyright (C) 2009 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.

// First added:  2009-09-25
// Last changed: 2009-09-28

// Template specializations of str for simple types

#include "MeshFunction.h"

namespace dolfin
{

//-----------------------------------------------------------------------------
template<> std::string MeshFunction<double>::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint i = 0; i < _size; i++)
      s << "  (" << _dim << ", " << i << "): " << _values[i] << std::endl;
  }
  else
    s << "<MeshFunction of topological dimension " << _dim << " containing " << _size << " values>";
  return s.str();
}
//-----------------------------------------------------------------------------
template<> std::string MeshFunction<dolfin::uint>::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint i = 0; i < _size; i++)
      s << "  (" << _dim << ", " << i << "): " << _values[i] << std::endl;
  }
  else
    s << "<MeshFuncton of topological dimension " << _dim << " containing " << _size << " values>";
  return s.str();
}
//-----------------------------------------------------------------------------

}
