// Copyright (C) 2003-2011 Anders Logg
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
// Modified by Garth N. Wells, 2011
//
// First added:  2003-02-26
// Last changed: 2011-09-24

#include <sstream>
#include <dolfin/parameter/Parameters.h>
#include "UniqueIdGenerator.h"
#include "Variable.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Variable::Variable()
  : _name("x"), _label("unnamed data"), unique_id(UniqueIdGenerator::id())

{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::Variable(const std::string name, const std::string label)
  : _name(name), _label(label), unique_id(UniqueIdGenerator::id())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::Variable(const Variable& variable) : _name(variable._name),
  _label(variable._label), unique_id(UniqueIdGenerator::id())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::~Variable()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Variable& Variable::operator=(const Variable& variable)
{
  // Assign everything but unique_id
  parameters = variable.parameters;
  _name = variable._name;
  _label = variable._label;

  return *this;
}
//-----------------------------------------------------------------------------
void Variable::rename(const std::string name, const std::string label)
{
  _name = name;
  _label = label;
}
//-----------------------------------------------------------------------------
std::string Variable::name() const
{
  return _name;
}
//-----------------------------------------------------------------------------
std::string Variable::label() const
{
  return _label;
}
//-----------------------------------------------------------------------------
std::string Variable::str(bool verbose) const
{
  std::stringstream s;
  s << "<DOLFIN object "
    << _name << " (" << _label << ")>";
  return s.str();
}
//-----------------------------------------------------------------------------
