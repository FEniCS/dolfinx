// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-26
// Last changed: 2009-05-09

#include <sstream>
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
  s << "Variable " << _name << " (" << _label << ")";
  return s.str();
}
//-----------------------------------------------------------------------------
