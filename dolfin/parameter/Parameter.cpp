// Copyright (C) 2009-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <sstream>
#include <boost/lexical_cast.hpp>
#include <dolfin/log/log.h>
#include "Parameter.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameter::Parameter(std::string key, const char* x)
  : _value(std::string(x)), _access_count(0), _change_count(0), _is_set(true),
    _key(key), _description("missing description")
{
  check_key(key);
}
//-----------------------------------------------------------------------------
Parameter::Parameter(std::string key, Type ptype)
  : _access_count(0), _change_count(0), _is_set(false), _key(key),
    _description("missing description")
{
  check_key(key);

  if (ptype == Type::Bool)
    _value = false;
  else if (ptype == Type::Int)
    _value = (int) 0;
  else if (ptype == Type::Float)
    _value = (double) 0.0;
  else if (ptype == Type::String)
    _value = std::string();
  else
  {
    dolfin_error("Parameter.cpp",
                 "add unset parameter",
                 "Type unknown");
  }
}
//-----------------------------------------------------------------------------
Parameter::~Parameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::string Parameter::key() const
{
  return _key;
}
//-----------------------------------------------------------------------------
std::string Parameter::description() const
{
  return _description;
}
//-----------------------------------------------------------------------------
bool Parameter::is_set() const
{
  return _is_set;
}
//-----------------------------------------------------------------------------
void Parameter::reset()
{
  _is_set = false;
}
//-----------------------------------------------------------------------------
std::size_t Parameter::access_count() const
{
  return _access_count;
}
//-----------------------------------------------------------------------------
std::size_t Parameter::change_count() const
{
  return _change_count;
}
//-----------------------------------------------------------------------------
void Parameter::set_range(int min_value, int max_value)
{
  if (_value.which() != 2)
  {
    dolfin_error("Parameter.cpp",
                 "set range for parameter",
                 "Cannot set int-valued range for parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }
  else
    _range = std::array<int, 2>({{min_value, max_value}});
}
//-----------------------------------------------------------------------------
void Parameter::set_range(double min_value, double max_value)
{
  if (_value.which() != 3)
  {
    dolfin_error("Parameter.cpp",
                 "set range for parameter",
                 "Cannot set double-valued range for parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }
  else
    _range = std::array<double, 2>({{min_value, max_value}});
}
//-----------------------------------------------------------------------------
void Parameter::set_range(std::set<std::string> range)
{
  if (_value.which() != 4)
  {
    dolfin_error("Parameter.cpp",
                 "set range for parameter",
                 "Cannot set string-valued range for parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }
  else
    _range = range;
}
//-----------------------------------------------------------------------------
void Parameter::get_range(int& min_value, int& max_value) const
{
  // FIXME: This is a workaround to support old (but bad) behaviour
  // where ranges were returned even when not set.
  if (_range.which() == 0)
  { min_value = 0; max_value = 0; return; }

  if (_value.which() != 2 or _range.which() == 0)
  {
    dolfin_error("Parameter.cpp",
                 "get range for parameter",
                 "Cannot get int-valued range for parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }

  min_value = boost::get<std::array<int, 2>>(_range)[0];
  max_value = boost::get<std::array<int, 2>>(_range)[1];
}
//-----------------------------------------------------------------------------
void Parameter::get_range(double& min_value, double& max_value) const
{
  // FIXME: This is a workaround to support old (but bad) behaviour
  // where ranges were returned even when not set.
  if (_range.which() == 0)
  { min_value = 0; max_value = 0; return; }

  if (_value.which() != 3 or _range.which() == 0)
  {
    dolfin_error("Parameter.cpp",
                 "get range for parameter",
                 "Cannot get double-valued range for parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }

  min_value = boost::get<std::array<double, 2>>(_range)[0];
  max_value = boost::get<std::array<double, 2>>(_range)[1];
}
//-----------------------------------------------------------------------------
void Parameter::get_range(std::set<std::string>& range) const
{
  // FIXME: This is a workaround to support old (but bad) behaviour
  // where ranges were returned even when not set.
  if (_range.which() == 0)
  { range = std::set<std::string>() ; return; }

  if (_value.which() != 4 or _range.which() == 0)
  {
    dolfin_error("Parameter.cpp",
                 "get range for parameter",
                 "Cannot get string-valued range for parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }

  range = boost::get<std::set<std::string>>(_range);
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (int value)
{
  if (_value.which() != 2)
  {
    dolfin_error("Parameter.cpp",
               "assign parameter",
               "Cannot assign int-value to parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
  }

  _value = value;
  _is_set = true;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (double value)
{
  if (_value.which() != 3)
  {
    dolfin_error("Parameter.cpp",
                 "assign parameter",
                 "Cannot assign double-value to parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }

  _value = value;
  _is_set = true;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (std::string value)
{
  if (_value.which() != 4)
  {
    dolfin_error("Parameter.cpp",
                 "assign parameter",
                 "Cannot assign string-value to parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }

  _value = value;
  _is_set = true;
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (const char* value)
{
  if (_value.which() != 4)
  {
    dolfin_error("Parameter.cpp",
                 "assign parameter",
                 "Cannot assign char-value to parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }

  _value = std::string(value);
  _is_set = true;
  return *this;
}
//-----------------------------------------------------------------------------
boost::variant<boost::blank, bool, int, double, std::string> Parameter::value() const
{
  return _value;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (bool value)
{
  if (_value.which() != 1)
  {
    dolfin_error("Parameter.cpp",
                 "assign parameter",
                 "Cannot assign bool-value to parameter \"%s\" of type %s",
                 _key.c_str(), type_str().c_str());
  }

  _value = value;
  _is_set = true;
  return *this;
}
//-----------------------------------------------------------------------------
Parameter::operator int() const
{
  if (_value.which() != 2)
  {
    dolfin_error("Parameter.cpp",
                 "convert to integer",
                 "Cannot convert parameter \"%s\" of type %s to int",
                 _key.c_str(), type_str().c_str());
  }

  return boost::get<int>(_value);
}
//-----------------------------------------------------------------------------
Parameter::operator std::size_t() const
{
  if (_value.which() != 2)
  {
    dolfin_error("Parameter.cpp",
                 "convert to unsigned integer",
                 "Cannot convert parameter \"%s\" of type %s to std::size_t",
                 _key.c_str(), type_str().c_str());
  }

  return boost::get<int>(_value);
}
//-----------------------------------------------------------------------------
Parameter::operator double() const
{
  if (_value.which() != 3)
  {
    dolfin_error("Parameter.cpp",
                 "convert to double",
                 "Cannot convert parameter \"%s\" of type %s to double",
                 _key.c_str(), type_str().c_str());
  }

  return boost::get<double>(_value);
}
//-----------------------------------------------------------------------------
Parameter::operator std::string() const
{
  if (_value.which() != 4)
  {
    dolfin_error("Parameter.cpp",
                 "convert to string",
                 "Cannot convert parameter \"%s\" of type %s to string",
                 _key.c_str(), type_str().c_str());
  }

  return boost::get<std::string>(_value);
}
//-----------------------------------------------------------------------------
Parameter::operator bool() const
{
  if (_value.which() != 1)
  {
    dolfin_error("Parameter.cpp",
                 "convert to string",
                 "Cannot convert parameter \"%s\" of type %s to bool",
                 _key.c_str(), type_str().c_str());
  }

  return boost::get<bool>(_value);
}
//-----------------------------------------------------------------------------
void Parameter::check_key(std::string key)
{
  // Space and punctuation not allowed in key names
  if (key.find(' ') != std::string::npos or key.find('.') != std::string::npos)
  {
    dolfin_error("Parameter.cpp",
                 "check allowed name for key",
                 "Illegal character in parameter key \"%s\" (no spaces for periods allowed)",
                 key.c_str());
  }
}
//-----------------------------------------------------------------------------
std::string Parameter::type_str() const
{
  switch (_value.which())
  {
  case 1:
    return "bool";
  case 2:
    return "int";
  case 3:
    return "double";
  case 4:
    return "string";
  }

  dolfin_error("Parameter.cpp",
               "return parameter type string",
               "Cannot determine parameter type");
  return "unknown";
}
//-----------------------------------------------------------------------------
std::string Parameter::value_str() const
{
  switch (_value.which())
  {
  case 1:
    return std::to_string(boost::get<bool>(_value));
  case 2:
    return std::to_string(boost::get<int>(_value));
  case 3:
    return std::to_string(boost::get<double>(_value));
  case 4:
    return boost::get<std::string>(_value);
  }

  dolfin_error("Parameter.cpp",
               "return parameter as string",
               "Cannot determine parameter type");
  return "unknown";
}
//-----------------------------------------------------------------------------
std::string Parameter::range_str() const
{
  if (_range.which() == 0)
    return "Not set";

  switch (_value.which())
  {
  case 1:
    return "{true, false}";
  case 2:
  {
    std::array<int, 2> ri = boost::get<std::array<int, 2>>(_range);
    return ("[" + std::to_string(ri[0]) + ", " + std::to_string(ri[1]) + "]");
  }
  case 3:
  {
    std::array<double, 2> rd = boost::get<std::array<double, 2>>(_range);
    return "[" + std::to_string(rd[0]) + ", " + std::to_string(rd[1]) + "]";
  }
  case 4:
  {
    std::set<std::string> _set = boost::get<std::set<std::string>>(_range);
    std::string rstr = "[";
    for (auto s : _set)
      rstr += s + ",";
    if (!_set.empty())
      rstr.pop_back();
    rstr += "]";
    return rstr;
  }
  }

  dolfin_error("Parameter.cpp",
               "return parameter as range",
               "Cannot determine parameter type");
  return "unknown";
}
//-----------------------------------------------------------------------------
std::string Parameter::str() const
{
  switch (_value.which())
  {
  case 1:
    return "<bool-valued parameter named \"" + key() + "\" with value "
      + boost::lexical_cast<std::string>(boost::get<bool>(_value)) + ">";
  case 2:
    return "<int-valued parameter named \"" + key() + "\" with value "
      + std::to_string(boost::get<int>(_value)) + ">";
  case 3:
    return "<double-valued parameter named \"" + key() + "\" with value "
      + std::to_string(boost::get<double>(_value)) + ">";
  case 4:
    return "<string-valued parameter named \"" + key() + "\" with value "
      + boost::get<std::string>(_value) + ">";
  }

  dolfin_error("Parameter.cpp",
               "return parameter as range",
               "Cannot determine parameter type");
  return "unknown";
}
//-----------------------------------------------------------------------------
