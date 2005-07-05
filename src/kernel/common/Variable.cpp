// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-26
// Last changed: 2005

#include <dolfin/Variable.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Variable::Variable() :
  _name("x"), _label("data with no label")
{
  _number = 0;
}
//-----------------------------------------------------------------------------
Variable::Variable(const std::string name, const std::string label) :
  _name(name), _label(label)
{
  _number = 0;
}
//-----------------------------------------------------------------------------
Variable::Variable(const Variable& variable) :
  _name(variable._name), _label(variable._label)
{
  _number = 0;
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
void Variable::operator++()
{
  _number++;
}
//-----------------------------------------------------------------------------
int Variable::number() const
{
  return _number;
}
//-----------------------------------------------------------------------------
