// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Variable.h"
#include "UniqueIdGenerator.h"
#include <dolfin/parameter/Parameters.h>
#include <sstream>

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
Variable::Variable(const Variable& variable)
    : _name(variable._name), _label(variable._label),
      unique_id(UniqueIdGenerator::id())
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
std::string Variable::name() const { return _name; }
//-----------------------------------------------------------------------------
std::string Variable::label() const { return _label; }
//-----------------------------------------------------------------------------
std::string Variable::str(bool verbose) const
{
  std::stringstream s;
  s << "<DOLFIN object " << _name << " (" << _label << ")>";
  return s.str();
}
//-----------------------------------------------------------------------------
