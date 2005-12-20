// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005-12-18

#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterValue.h>
#include <dolfin/Parameter.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameter::Parameter(int value) : value(0), _type(type_int)
{
  this->value = new IntValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(real value) : value(0), _type(type_real)
{
  this->value = new RealValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(bool value) : value(0), _type(type_bool)
{
  this->value = new BoolValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(std::string value) : value(0), _type(type_string)
{
  this->value = new StringValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(uint value) : value(0), _type(type_int)
{
  this->value = new IntValue(static_cast<int>(value));
}
//-----------------------------------------------------------------------------
Parameter::Parameter(const Parameter& parameter)
  : value(0), _type(parameter._type)
{
  switch ( parameter._type )
  {
  case type_int:
    value = new IntValue(*parameter.value);
    break;
  case type_real:
    value = new RealValue(*parameter.value);
    break;
  case type_bool:
    value = new BoolValue(*parameter.value);
    break;
  case type_string:
    value = new StringValue(*parameter.value);
    break;
  default:
    dolfin_error1("Unknown parameter type: %d.", parameter._type);
  }
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (int value)
{
  *(this->value) = value;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (real value)
{
  *(this->value) = value;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (bool value)
{
  *(this->value) = value;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (std::string value)
{
  *(this->value) = value;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (uint value)
{
  *(this->value) = value;
  return *this;
}
//-----------------------------------------------------------------------------
Parameter::~Parameter()
{
  if ( value ) delete value;
}
//-----------------------------------------------------------------------------
Parameter::operator int() const
{
  return *value;
}
//-----------------------------------------------------------------------------
Parameter::operator real() const
{
  return *value;
}
//-----------------------------------------------------------------------------
Parameter::operator bool() const
{
  return *value;
}
//-----------------------------------------------------------------------------
Parameter::operator std::string() const
{
  return *value;
}
//-----------------------------------------------------------------------------
Parameter::operator uint() const
{
  return *value;
}
//-----------------------------------------------------------------------------
Parameter::Type Parameter::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
