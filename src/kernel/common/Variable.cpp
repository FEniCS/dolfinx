// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Variable.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Variable::Variable() :
  _name("x"), _label("data with no label")
{

}
//-----------------------------------------------------------------------------
Variable::Variable(const std::string name, const std::string label) :
  _name(name), _label(label)
{

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
