// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-08-09
// Last changed: 2010-11-12

#include <boost/functional/hash.hpp>
#include <cstdlib>
#include <sstream>
#include "types.h"
#include "utils.h"

//-----------------------------------------------------------------------------
std::string dolfin::indent(std::string block)
{
  std::string indentation("  ");
  std::stringstream s;

  s << indentation;
  for (uint i = 0; i < block.size(); ++i)
  {
    s << block[i];
    if (block[i] == '\n' && i < block.size() - 1)
      s << indentation;
  }

  return s.str();
}
//-----------------------------------------------------------------------------
std::string dolfin::to_string(int n)
{
  std::stringstream s;
  s << n;
  return s.str();
}
//-----------------------------------------------------------------------------
std::string dolfin::to_string(double x)
{
  std::stringstream s;
  s << x;
  return s.str();
}
//-----------------------------------------------------------------------------
std::string dolfin::to_string(const double* x, uint n)
{
  std::stringstream s;

  s << "[";
  for (uint i = 0; i < n; i++)
  {
    s << x[i];
    if (i < n - 1)
      s << ", ";
  }
  s << "]";

  return s.str();
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::hash(std::string signature)
{
  boost::hash<std::string> string_hash;
  std::size_t h = string_hash(signature);

  return h;
}
//-----------------------------------------------------------------------------
