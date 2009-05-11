// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-26
// Last changed: 2009-05-11

#include <dolfin/log/dolfin_log.h>

#include <sstream>
#include <dolfin/parameter/NewParameters.h>
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
std::string Variable::str() const
{
  std::stringstream s;
  s << _name << " (" << _label << ")";
  return s.str();
}
//-----------------------------------------------------------------------------
NewParameters Variable::default_parameters() const
{
  // Return empty parameter database if not overloaded by subclass.
  // Note that although this method is overloaded by a subclass, the
  // overloaded method will not be called by the Variable base class
  // constructor since that call happens before the constructor of the
  // subclass. Instead, subclasses must explicitly assign to the
  // parameters variable in their constructors.

  NewParameters parameters("Default parameters, not overloaded by subclass");
  return parameters;
}
//-----------------------------------------------------------------------------
