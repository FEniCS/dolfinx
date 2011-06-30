// Copyright (C) 2009 Ola Skavhaug
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added: 2009-03-17
// Last changed: 2009-03-17

#include <iomanip>
#include "XMLIndent.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLIndent::XMLIndent(uint indentation_level, uint step_size)
  : indentation_level(indentation_level), step_size(step_size)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLIndent::~XMLIndent()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLIndent::operator++()
{
  ++indentation_level;
}
//-----------------------------------------------------------------------------
void XMLIndent::operator--()
{
  -- indentation_level;
}
//-----------------------------------------------------------------------------
std::string XMLIndent::operator()()
{
  std::ostringstream ss;
  ss << std::setw(indentation_level*step_size) << "";
  return ss.str();
}
//-----------------------------------------------------------------------------
