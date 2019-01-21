// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Variable.h"
#include "UniqueIdGenerator.h"
#include <sstream>

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------------
Variable::Variable() : _name("x"), unique_id(common::UniqueIdGenerator::id())

{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::Variable(const std::string name)
    : _name(name), unique_id(common::UniqueIdGenerator::id())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::Variable(const Variable& variable)
    : _name(variable._name), unique_id(common::UniqueIdGenerator::id())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Variable& Variable::operator=(const Variable& variable)
{
  // Assign everything but unique_id
  _name = variable._name;

  return *this;
}
//-----------------------------------------------------------------------------
void Variable::rename(const std::string name) { _name = name; }
//-----------------------------------------------------------------------------
std::string Variable::name() const { return _name; }
//-----------------------------------------------------------------------------
std::string Variable::str(bool verbose) const
{
  std::stringstream s;
  s << "<DOLFIN object " << _name << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
