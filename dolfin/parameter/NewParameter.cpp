// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-05-08
// Last changed: 2009-05-08

#include <sstream>
#include <dolfin/log/log.h>
#include "NewParameter.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// class Parameter
//-----------------------------------------------------------------------------
NewParameter::NewParameter(std::string key)
  : _access_count(0), _change_count(0),
    _key(key), _description("missing description")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewParameter::~NewParameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::string NewParameter::key() const
{
  return _key;
}
//-----------------------------------------------------------------------------
std::string NewParameter::description() const
{
  return _description;
}
//-----------------------------------------------------------------------------
dolfin::uint NewParameter::access_count() const
{
  return _access_count;
}
//-----------------------------------------------------------------------------
dolfin::uint NewParameter::change_count() const
{
  return _change_count;
}
//-----------------------------------------------------------------------------
void NewParameter::set_range(int min_value, int max_value)
{
  error("Cannot set double-valued range for parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void NewParameter::set_range(double min_value, double max_value)
{
  error("Cannot set int-valued range for parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void NewParameter::set_range(const std::set<std::string>& range)
{
  error("Cannot set string-valued range for parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
const NewParameter& NewParameter::operator= (int value)
{
  error("Cannot assign int-value to parameter of type %s.",
        type_str().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const NewParameter& NewParameter::operator= (double value)
{
  error("Cannot assign double-value to parameter of type %s.",
        type_str().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const NewParameter& NewParameter::operator= (std::string value)
{
  error("Cannot assign string-value to parameter of type %s.",
        type_str().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
NewParameter::operator int() const
{
  error("Unable to convert parameter of type %s to int.", type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
NewParameter::operator double() const
{
  error("Unable to convert parameter of type %s to double.", type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
NewParameter::operator std::string() const
{
  error("Unable to convert parameter of type %s to string.", type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
// class IntParameter
//-----------------------------------------------------------------------------
NewIntParameter::NewIntParameter(std::string key, int value)
  : NewParameter(key), _value(value), _min(0), _max(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewIntParameter::~NewIntParameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewIntParameter::set_range(int min_value, int max_value)
{
  // Check range
  if (min_value > max_value)
    error("Illegal range for int-valued parameter: [%d, %d].",
          min_value, max_value);

  // Set range
  _min = min_value;
  _max = max_value;
}
//-----------------------------------------------------------------------------
const NewIntParameter& NewIntParameter::operator= (int value)
{
  // Check value
  if (_min != _max && (value < _min || value > _max))
    error("Parameter value %d for parameter \"%s\" out of range [%d, %d].",
          value, key().c_str(), _min, _max);

  // Set value
  _value = value;
  _change_count++;

  return *this;
}
//-----------------------------------------------------------------------------
NewIntParameter::operator int() const
{
  _access_count++;
  return _value;
}
//-----------------------------------------------------------------------------
std::string NewIntParameter::type_str() const
{
  return "int";
}
//-----------------------------------------------------------------------------
std::string NewIntParameter::value_str() const
{
  std::stringstream s;
  s << _value;
  return s.str();
}
//-----------------------------------------------------------------------------
std::string NewIntParameter::range_str() const
{
  std::stringstream s;
  if (_min == _max)
    s << "[]";
  else
    s << "[" << _min << ", " << _max << "]";
  return s.str();
}
//-----------------------------------------------------------------------------
std::string NewIntParameter::str() const
{
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
NewDoubleParameter::NewDoubleParameter(std::string key, double value)
  : NewParameter(key), _value(value), _min(0), _max(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewDoubleParameter::~NewDoubleParameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewDoubleParameter::set_range(double min_value, double max_value)
{
  // Check range
  if (min_value > max_value)
    error("Illegal range for double-valued parameter: [%g, %g].",
          min_value, max_value);

  // Set range
  _min = min_value;
  _max = max_value;
}
//-----------------------------------------------------------------------------
const NewDoubleParameter& NewDoubleParameter::operator= (double value)
{
  // Check value
  if (_min != _max && (value < _min || value > _max))
    error("Parameter value %g for parameter \"%s\" out of range [%g, %g].",
          value, key().c_str(), _min, _max);

  // Set value
  _value = value;
  _change_count++;

  return *this;
}
//-----------------------------------------------------------------------------
NewDoubleParameter::operator double() const
{
  _access_count++;
  return _value;
}
//-----------------------------------------------------------------------------
std::string NewDoubleParameter::type_str() const
{
  return "double";
}
//-----------------------------------------------------------------------------
std::string NewDoubleParameter::value_str() const
{
  std::stringstream s;
  s << _value;
  return s.str();
}
//-----------------------------------------------------------------------------
std::string NewDoubleParameter::range_str() const
{
  std::stringstream s;
  if (_min == _max)
    s << "[]";
  else
    s << "[" << _min << ", " << _max << "]";
  return s.str();
}
//-----------------------------------------------------------------------------
std::string NewDoubleParameter::str() const
{
  std::stringstream s;
  s << "<double-valued parameter named \""
    << key()
    << "\" with value "
    << _value
    << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
NewStringParameter::NewStringParameter(std::string key, std::string value)
  : NewParameter(key), _value(value)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewStringParameter::~NewStringParameter()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewStringParameter::set_range(const std::set<std::string>& range)
{
  _range = range;
}
//-----------------------------------------------------------------------------
const NewStringParameter& NewStringParameter::operator= (std::string value)
{
  // Check value
  if (_range.size() > 0 && _range.find(value) == _range.end())
  {
    std::stringstream s;
    s << "Illegal value for parameter. Allowed values are: " << range_str();
    error(s.str());
  }

  // Set value
  _value = value;
  _change_count++;

  return *this;
}
//-----------------------------------------------------------------------------
NewStringParameter::operator std::string() const
{
  _access_count++;
  return _value;
}
//-----------------------------------------------------------------------------
std::string NewStringParameter::type_str() const
{
  return "string";
}
//-----------------------------------------------------------------------------
std::string NewStringParameter::value_str() const
{
  return _value;
}
//-----------------------------------------------------------------------------
std::string NewStringParameter::range_str() const
{
  std::stringstream s;
  s << "[";
  uint i = 0;
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
std::string NewStringParameter::str() const
{
  std::stringstream s;
  s << "<string-valued parameter named \""
    << key()
    << "\" with value "
    << _value
    << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
