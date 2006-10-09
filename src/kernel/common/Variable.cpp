// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-26
// Last changed: 2006-10-09

#include <dolfin/Variable.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Variable::Variable() :
  _name("x"), _label("data with no label")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::Variable(const std::string name, const std::string label) :
  _name(name), _label(label)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Variable::Variable(const Variable& variable) :
  _name(variable._name), _label(variable._label)
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
