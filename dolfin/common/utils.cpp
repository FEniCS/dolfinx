// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-08-09
// Last changed: 2010-11-18

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
