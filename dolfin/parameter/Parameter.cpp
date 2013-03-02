// Copyright (C) 2009-2011 Anders Logg
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
//
// Modified by Marie Rognes 2011
// Modified by Joachim B Haga 2012
//
// First added:  2009-05-08
// Last changed: 2012-09-11

#include <sstream>
#include <dolfin/log/log.h>
#include "Parameter.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// class Parameter
//-----------------------------------------------------------------------------
Parameter::Parameter(std::string key)
  : _access_count(0), _change_count(0), _is_set(false),
    _key(key), _description("missing description")
{
  // Check that key name is allowed
  check_key(key);
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
  dolfin_error("Parameter.cpp",
               "set range for parameter",
               "Cannot set int-valued range for parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
}
//-----------------------------------------------------------------------------
void Parameter::set_range(double min_value, double max_value)
{
  dolfin_error("Parameter.cpp",
               "set range for parameter",
               "Cannot set double-valued range for parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
}
//-----------------------------------------------------------------------------
void Parameter::set_range(std::set<std::string> range)
{
  dolfin_error("Parameter.cpp",
               "set range for parameter",
               "Cannot set string-valued range for parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
}
//-----------------------------------------------------------------------------
void Parameter::get_range(int& min_value, int& max_value) const
{
  dolfin_error("Parameter.cpp",
               "get range for parameter",
               "Cannot get int-valued range for parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
}
//-----------------------------------------------------------------------------
void Parameter::get_range(double& min_value, double& max_value) const
{
  dolfin_error("Parameter.cpp",
               "get range for parameter",
               "Cannot get double-valued range for parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
}
//-----------------------------------------------------------------------------
void Parameter::get_range(std::set<std::string>& range) const
{
  dolfin_error("Parameter.cpp",
               "get range for parameter",
               "Cannot get string-valued range for parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (int value)
{
  dolfin_error("Parameter.cpp",
               "assign parameter",
               "Cannot assign int-value to parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (double value)
{
  dolfin_error("Parameter.cpp",
               "assign parameter",
               "Cannot assign double-value to parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (std::string value)
{
  dolfin_error("Parameter.cpp",
               "assign parameter",
               "Cannot assign string-value to parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (const char* value)
{
  dolfin_error("Parameter.cpp",
               "assign parameter",
               "Cannot assign char-value to parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const Parameter& Parameter::operator= (bool value)
{
  dolfin_error("Parameter.cpp",
               "assign parameter",
               "Cannot assign bool-value to parameter \"%s\" of type %s",
               _key.c_str(), type_str().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
Parameter::operator int() const
{
  dolfin_error("Parameter.cpp",
               "convert to integer",
               "Cannot convert parameter \"%s\" of type %s to int",
               _key.c_str(), type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
Parameter::operator std::size_t() const
{
  dolfin_error("Parameter.cpp",
               "convert to unsigned integer",
               "Cannot convert parameter \"%s\" of type %s to std::size_t",
               _key.c_str(), type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
Parameter::operator double() const
{
  dolfin_error("Parameter.cpp",
               "convert to double",
               "Cannot convert parameter \"%s\" of type %s to double",
               _key.c_str(), type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
Parameter::operator std::string() const
{
  dolfin_error("Parameter.cpp",
               "convert to string",
               "Cannot convert parameter \"%s\" of type %s to string",
               _key.c_str(), type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
Parameter::operator bool() const
{
  dolfin_error("Parameter.cpp",
               "convert to string",
               "Cannot convert parameter \"%s\" of type %s to bool",
               _key.c_str(), type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
void Parameter::check_key(std::string key)
{
  // Space and punctuation not allowed in key names
  for (std::size_t i = 0; i < key.size(); i++)
  {
    if (key[i] == ' ' || key[i] == '.')
    {
      dolfin_error("Parameter.cpp",
                   "check allowed name for key",
                   "Illegal character '%c' in parameter key \"%s\"",
                   key[i], key.c_str());
    }
  }
}
//-----------------------------------------------------------------------------
// class IntParameter
//-----------------------------------------------------------------------------
IntParameter::IntParameter(std::string key)
  : Parameter(key), _min(0), _max(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IntParameter::IntParameter(std::string key, int value)
  : Parameter(key), _value(value), _min(0), _max(0)
{
  _is_set = true;
}
//-----------------------------------------------------------------------------
IntParameter::~IntParameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void IntParameter::set_range(int min_value, int max_value)
{
  // Check range
  if (min_value > max_value)
  {
    dolfin_error("Parameter.cpp",
                 "set range for parameter",
                 "Illegal range for int-valued parameter: [%d, %d]",
                 min_value, max_value);
  }

  // Set range
  _min  = min_value;
  _max = max_value;
}
//-----------------------------------------------------------------------------
void IntParameter::get_range(int& min_value, int& max_value) const
{
  // Get range
  min_value = _min;
  max_value = _max;
}
//-----------------------------------------------------------------------------
const IntParameter& IntParameter::operator= (int value)
{
  // Check value
  if (_min != _max && (value < _min || value > _max))
  {
    dolfin_error("Parameter.cpp",
                 "assign value to parameter",
                 "Value %d out of allowed range [%d, %d] for parameter\"%s\"",
                 value, _min, _max, key().c_str());
  }

  // Set value
  _value = value;
  _change_count++;
  _is_set = true;

  return *this;
}
//-----------------------------------------------------------------------------
IntParameter::operator int() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "convert parameter to int",
                 "Parameter has not been set");
  }

  _access_count++;
  return _value;
}
//-----------------------------------------------------------------------------
IntParameter::operator std::size_t() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "convert int parameter to std::size_t",
                 "Parameter has not been set");
  }

  if (_value < 0)
  {
    dolfin_error("Parameter.cpp",
                 "convert int parameter to std::size_t",
                 "Parameter \"%s\" has negative value %d",
                 key().c_str(), _value);
  }

  _access_count++;
  return _value;
}
//-----------------------------------------------------------------------------
std::string IntParameter::type_str() const
{
  return "int";
}
//-----------------------------------------------------------------------------
std::string IntParameter::value_str() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "get string representation of value",
                 "Parameter has not been set");
  }

  std::stringstream s;
  s << _value;
  return s.str();
}
//-----------------------------------------------------------------------------
std::string IntParameter::range_str() const
{
  std::stringstream s;
  if (_min == _max)
    s << "[]";
  else
    s << "[" << _min << ", " << _max << "]";
  return s.str();
}
//-----------------------------------------------------------------------------
std::string IntParameter::str() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "get string representation of parameter",
                 "Parameter has not been set");
  }

  std::stringstream s;
  s << "<int-valued parameter named \""
    << key()
    << "\" with value "
    << _value
    << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
// class DoubleParameter
//-----------------------------------------------------------------------------
DoubleParameter::DoubleParameter(std::string key)
  : Parameter(key), _min(0.0), _max(0.0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DoubleParameter::DoubleParameter(std::string key, double value)
  : Parameter(key), _value(value), _min(0.0), _max(0.0)
{
  _is_set = true;
}
//-----------------------------------------------------------------------------
DoubleParameter::~DoubleParameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DoubleParameter::set_range(double min_value, double max_value)
{
  // Check range
  if (min_value > max_value)
  {
    dolfin_error("Parameter.cpp",
                 "set range for parameter",
                 "Illegal range for double-valued parameter: [%g, %g]",
                 min_value, max_value);
  }

  // Set range
  _min = min_value;
  _max = max_value;
}
//-----------------------------------------------------------------------------
void DoubleParameter::get_range(double& min_value, double& max_value) const
{
  // Get range
  min_value = _min;
  max_value = _max;
}
//-----------------------------------------------------------------------------
const DoubleParameter& DoubleParameter::operator= (double value)
{
  // Check value
  if (_min != _max && (value < _min || value > _max))
  {
    dolfin_error("Parameter.cpp",
                 "assign value to parameter",
                 "Value %g out of allowed range [%g, %g] for parameter\"%s\"",
                 value, _min, _max, key().c_str());
  }

  // Set value
  _value = value;
  _change_count++;
  _is_set = true;

  return *this;
}
//-----------------------------------------------------------------------------
DoubleParameter::operator double() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "convert parameter to double",
                 "Parameter has not been set");
  }

  _access_count++;
  return _value;
}
//-----------------------------------------------------------------------------
std::string DoubleParameter::type_str() const
{
  return "double";
}
//-----------------------------------------------------------------------------
std::string DoubleParameter::value_str() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "get string representation of value",
                 "Parameter has not been set");
  }

  std::stringstream s;
  s << _value;
  return s.str();
}
//-----------------------------------------------------------------------------
std::string DoubleParameter::range_str() const
{
  std::stringstream s;
  if (_min == _max)
    s << "[]";
  else
    s << "[" << _min << ", " << _max << "]";
  return s.str();
}
//-----------------------------------------------------------------------------
std::string DoubleParameter::str() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "get string representation of parameter",
                 "Parameter has not been set");
  }

  std::stringstream s;
  s << "<double-valued parameter named \""
    << key()
    << "\" with value "
    << _value
    << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
// class StringParameter
//-----------------------------------------------------------------------------
StringParameter::StringParameter(std::string key) : Parameter(key)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
StringParameter::StringParameter(std::string key, std::string value)
  : Parameter(key), _value(value)
{
  _is_set = true;
}
//-----------------------------------------------------------------------------
StringParameter::~StringParameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void StringParameter::set_range(std::set<std::string> range)
{
  _range = range;
}
//-----------------------------------------------------------------------------
void StringParameter::get_range(std::set<std::string>& range) const
{
  // Get range
  range = _range;
}
//-----------------------------------------------------------------------------
const StringParameter& StringParameter::operator= (std::string value)
{
  // Check value
  if (!_range.empty() && _range.find(value) == _range.end())
  {
    std::stringstream s;
    s << "Illegal value for parameter. Allowed values are: " << range_str();
    dolfin_error("Parameter.cpp",
                 "assign parameter value",
                 s.str());
  }

  // Set value
  _value = value;
  _change_count++;
  _is_set = true;

  return *this;
}
//-----------------------------------------------------------------------------
const StringParameter& StringParameter::operator= (const char* value)
{
  std::string s(value);

  // Check value
  if (!_range.empty() && _range.find(s) == _range.end())
  {
    std::stringstream stream;
    stream << "Illegal value for parameter. Allowed values are: " << range_str();
    dolfin_error("Parameter.cpp",
                 "assign parameter value",
                 stream.str());
  }

  // Set value
  _value = s;
  _change_count++;
  _is_set = true;

  return *this;
}
//-----------------------------------------------------------------------------
StringParameter::operator std::string() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "convert parameter to string ",
                 "Parameter has not been set");
  }

  _access_count++;
  return _value;
}
//-----------------------------------------------------------------------------
std::string StringParameter::type_str() const
{
  return "string";
}
//-----------------------------------------------------------------------------
std::string StringParameter::value_str() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "get string representation of value",
                 "Parameter has not been set");
  }
  return _value;
}
//-----------------------------------------------------------------------------
std::string StringParameter::range_str() const
{
  std::stringstream s;
  s << "[";
  std::size_t i = 0;
  for (std::set<std::string>::const_iterator it = _range.begin();
       it != _range.end(); ++it)
  {
    s << *it;
    if (i++ < _range.size() - 1)
      s << ", ";
  }
  s << "]";

  return s.str();
}
//-----------------------------------------------------------------------------
std::string StringParameter::str() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "get string representation of parameter",
                 "Parameter has not been set");
  }

  std::stringstream s;
  s << "<string-valued parameter named \""
    << key()
    << "\" with value "
    << _value
    << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
// class BoolParameter
//-----------------------------------------------------------------------------
BoolParameter::BoolParameter(std::string key) : Parameter(key)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoolParameter::BoolParameter(std::string key, bool value)
  : Parameter(key), _value(value)
{
  _is_set = true;
}
//-----------------------------------------------------------------------------
BoolParameter::~BoolParameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const BoolParameter& BoolParameter::operator= (bool value)
{
  // Set value
  _value = value;
  _change_count++;
  _is_set = true;

  return *this;
}
//-----------------------------------------------------------------------------
BoolParameter::operator bool() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "convert parameter to bool",
                 "Parameter has not been set");
  }

  _access_count++;
  return _value;
}
//-----------------------------------------------------------------------------
std::string BoolParameter::type_str() const
{
  return "bool";
}
//-----------------------------------------------------------------------------
std::string BoolParameter::value_str() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "get string representation of value",
                 "Parameter has not been set");
  }

  if (_value)
    return "true";
  else
    return "false";
}
//-----------------------------------------------------------------------------
std::string BoolParameter::range_str() const
{
  return "{true, false}";
}
//-----------------------------------------------------------------------------
std::string BoolParameter::str() const
{
  if (!_is_set)
  {
    dolfin_error("Parameter.cpp",
                 "get string representation of parameter",
                 "Parameter has not been set");
  }

  std::stringstream s;
  s << "<bool-valued parameter named \""
    << key()
    << "\" with value "
    << _value
    << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
