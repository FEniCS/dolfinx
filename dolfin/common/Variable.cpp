// Copyright (C) 2003-2009 Anders Logg
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
// First added:  2003-02-26
// Last changed: 2009-08-11

#include <dolfin/log/dolfin_log.h>

#include <sstream>
#include <dolfin/parameter/Parameters.h>
#include "Variable.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Variable::Variable()
  : _name("x"), _label("unnamed data")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::Variable(const std::string name, const std::string label)
  : _name(name), _label(label)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::~Variable()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Variable::rename(const std::string name, const std::string label)
{
  _name = name;
  _label = label;
}
//-----------------------------------------------------------------------------
const std::string& Variable::name() const
{
  return _name;
}
//-----------------------------------------------------------------------------
const std::string& Variable::label() const
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
void Variable::disp() const
{
  warning("disp() function is deprecated, use info(foo) or info(foo, true) instead.");
}
//-----------------------------------------------------------------------------
