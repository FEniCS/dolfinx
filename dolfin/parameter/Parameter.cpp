// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet
//
// First added:  2003-05-06
// Last changed: 2009-05-06

#include <dolfin/log/dolfin_log.h>
#include "ParameterValue.h"
#include "Parameter.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameter::Parameter(int value)
  : _value(0), _type(type_int), _changed(false)
{
  _value = new IntValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(uint value)
  : _value(0), _type(type_int), _changed(false)
{
  _value = new IntValue(static_cast<int>(value));
}
//-----------------------------------------------------------------------------
Parameter::Parameter(double value)
  : _value(0), _type(type_double), _changed(false)
{
  _value = new DoubleValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(bool value)
  : _value(0), _type(type_bool), _changed(false)
{
  _value = new BoolValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(std::string value)
  : _value(0), _type(type_string), _changed(false)
{
  _value = new StringValue(value);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(const char* value)
  : _value(0), _type(type_string), _changed(false)
{
  std::string s(value);
  _value = new StringValue(s);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(const Parameter& parameter)
  : _value(0), _type(type_int), _changed(false)
{
  *this = parameter;
}
//-----------------------------------------------------------------------------
Parameter::~Parameter()
{
  delete _value;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (int value)
{
  _changed = true;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (dolfin::uint value)
{
  _changed = true;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (double value)
{
  _changed = true;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (bool value)
{
  _changed = true;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (std::string value)
{
  _changed = true;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (const Parameter& parameter)
{
  delete _value;

  switch (parameter._type)
  {
  case type_int:
    _value = new IntValue(*parameter._value);
    break;
  case type_double:
    _value = new DoubleValue(*parameter._value);
    break;
  case type_bool:
    _value = new BoolValue(*parameter._value);
    break;
  case type_string:
    _value = new StringValue(*parameter._value);
    break;
  default:
    error("Unknown parameter type: %d.", parameter._type);
  }

  _type = parameter._type;
  _changed = parameter._changed;

  return *this;
}
//-----------------------------------------------------------------------------
void Parameter::set_range(int min_value, int max_value)
{
  _value->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void Parameter::set_range(uint min_value, uint max_value)
{
  _value->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void Parameter::set_range(double min_value, double max_value)
{
  _value->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void Parameter::set_range(const std::vector<std::string>& allowed_values)
{
  _value->set_range(allowed_values);
}
//-----------------------------------------------------------------------------
Parameter::operator int() const
{
  return *_value;
}
//-----------------------------------------------------------------------------
Parameter::operator dolfin::uint() const
{
  return *_value;
}
//-----------------------------------------------------------------------------
Parameter::operator double() const
{
  return *_value;
}
//-----------------------------------------------------------------------------
Parameter::operator bool() const
{
  return *_value;
}
//-----------------------------------------------------------------------------
Parameter::operator std::string() const
{
  return *_value;
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
  switch (parameter.type())
  {
  case Parameter::type_int:
    stream << "[Parameter: value = "
	   << static_cast<int>(parameter) << " (int)]";
    break;
  case Parameter::type_double:
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
