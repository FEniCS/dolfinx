// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-05-06
// Last changed: 2005-12-21

#include <dolfin/log/dolfin_log.h>
#include "ParameterValue.h"
#include "Parameter.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameter::Parameter(int value) : value(0), _type(type_int)
{
  this->value = new IntValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(uint value) : value(0), _type(type_int)
{
  this->value = new IntValue(static_cast<int>(value));
}
//-----------------------------------------------------------------------------
Parameter::Parameter(double value) : value(0), _type(type_real)
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
Parameter::Parameter(const char* value) : value(0), _type(type_string)
{
  std::string s(value);
  this->value = new StringValue(s);
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
    error("Unknown parameter type: %d.", parameter._type);
  }
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (int value)
{
  *(this->value) = value;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (dolfin::uint value)
{
  *(this->value) = value;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (double value)
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
const Parameter& Parameter::operator= (const Parameter& parameter)
{
  delete value;

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
    error("Unknown parameter type: %d.", parameter._type);
  }  

  _type = parameter._type;

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
Parameter::operator dolfin::uint() const
{
  return *value;
}
//-----------------------------------------------------------------------------
Parameter::operator double() const
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
Parameter::Type Parameter::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream,
				      const Parameter& parameter)
{
  switch ( parameter.type() )
  {
  case Parameter::type_int:
    stream << "[Parameter: value = " 
	   << static_cast<int>(parameter) << " (int)]";
    break;
  case Parameter::type_real:
    stream << "[Parameter: value = "
	   << static_cast<double>(parameter) << " (double)]";
    break;
  case Parameter::type_bool:
    if ( static_cast<bool>(parameter) )
      stream << "[Parameter: value = true (bool)]";
    else
      stream << "[Parameter: value = false (bool)]";
    break;
  case Parameter::type_string:
    stream << "[Parameter: value = \""
	   << static_cast<std::string>(parameter) << "\" (string)]";
    break;
  default:
    error("Unknown parameter type: %d.", parameter._type);
  }
  
  return stream;
}
//-----------------------------------------------------------------------------
