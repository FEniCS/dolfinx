// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-18
// Last changed: 2009-05-06

#include <dolfin/log/dolfin_log.h>
#include "ParameterValue.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ParameterValue::ParameterValue()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ParameterValue::~ParameterValue()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ParameterValue::set(int value)
{
  error("Cannot assign int value to parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void ParameterValue::set(double value)
{
  error("Cannot assign double value to parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void ParameterValue::set(bool value)
{
  error("Cannot assign bool value to parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void ParameterValue::set(std::string value)
{
  error("Cannot assign string value to parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void ParameterValue::set_range(int min_value, int max_value)
{
  error("Cannot set int-valued range for parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void ParameterValue::set_range(uint min_value, uint max_value)
{
  error("Cannot set uint-valued range for parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void ParameterValue::set_range(double min_value, double max_value)
{
  error("Cannot set double-valued range for parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void ParameterValue::set_range(const std::vector<std::string>& allowed_values)
{
  error("Cannot string-valued range (allowed values) for parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
void ParameterValue::set(uint value)
{
  error("Cannot assign uint value to parameter of type %s.",
        type_str().c_str());
}
//-----------------------------------------------------------------------------
ParameterValue::operator int() const
{
  error("Unable to convert parameter of type %s to int.",
        type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
ParameterValue::operator double() const
{
  error("Unable to convert parameter of type %s to real.",
        type_str().c_str());
  return 0.0;
}
//-----------------------------------------------------------------------------
ParameterValue::operator bool() const
{
  error("Unable to convert parameter of type %s to bool.",
        type_str().c_str());
  return false;
}
//-----------------------------------------------------------------------------
ParameterValue::operator std::string() const
{
  error("Unable to convert parameter of type %s to string.",
        type_str().c_str());
  return "";
}
//-----------------------------------------------------------------------------
ParameterValue::operator dolfin::uint() const
{
  error("Unable to convert parameter of type %s to uint.",
        type_str().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
IntValue::IntValue(int value) 
  : ParameterValue(), value(value), min_value(0), max_value(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IntValue::~IntValue()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void IntValue::set(int value)
{ 
  // Check value
  if (min_value != max_value && (value < min_value || value > max_value))
  {
    error("Parameter value %d out of range [%d, %d].",
          value, min_value, max_value);
  }

  // Set value
  this->value = value;
}
//-----------------------------------------------------------------------------
void IntValue::set(uint value)
{ 
  // Check value
  int int_value = static_cast<int>(value);
  if (min_value != max_value && (int_value < min_value || int_value > max_value))
  {
    error("Parameter value %d out of range [%d, %d].",
          value, min_value, max_value);
  }

  // Set value
  this->value = int_value;
}
//-----------------------------------------------------------------------------
void IntValue::set_range(int min_value, int max_value)
{
  // Check range
  if (min_value > max_value)
    error("Illegal range for int-valued parameter: [%d, %d].",
          min_value, max_value);

  // Set range
  this->min_value = min_value;
  this->max_value = max_value;
}
//-----------------------------------------------------------------------------
IntValue::operator int() const
{
  return value; 
}
//-----------------------------------------------------------------------------
IntValue::operator uint() const
{
  if (value < 0)
    error("Unable to convert negative int parameter to uint.");
  return static_cast<uint>(value);
}
//-----------------------------------------------------------------------------
std::string IntValue::type_str() const
{
  return "int";
}
//-----------------------------------------------------------------------------
DoubleValue::DoubleValue(double value)
  : ParameterValue(), value(value), min_value(0), max_value(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DoubleValue::~DoubleValue()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DoubleValue::set(double value)
{ 
  if (min_value != max_value && (value < min_value || value > max_value))
  {
    error("Parameter value %g out of range [%g, %g].",
          value, min_value, max_value);
  }       
  this->value = value;
}
//-----------------------------------------------------------------------------
void DoubleValue::set_range(double min_value, double max_value)
{
  // Check range
  if (min_value > max_value)
    error("Illegal range for double-valued parameter: [%g, %g].",
          min_value, max_value);

  // Set range
  this->min_value = min_value;
  this->max_value = max_value;
}
//-----------------------------------------------------------------------------
DoubleValue::operator double() const
{
  return value;
}
//-----------------------------------------------------------------------------
std::string DoubleValue::type_str() const
{ 
  return "real";
}
//-----------------------------------------------------------------------------
BoolValue::BoolValue(bool value) : ParameterValue(), value(value)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoolValue::~BoolValue()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoolValue::set(bool value)
{
  this->value = value;
}
//-----------------------------------------------------------------------------
BoolValue::operator bool() const
{
  return value;
}
//-----------------------------------------------------------------------------
std::string BoolValue::type_str() const
{
  return "bool";
}
//-----------------------------------------------------------------------------
StringValue::StringValue(std::string value) : ParameterValue(), value(value)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
StringValue::~StringValue()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void StringValue::set(std::string value)
{ 
  // Check value
  if (allowed_values.size() > 0)
  {
    bool ok = false;
    for (uint i = 0; i < allowed_values.size(); ++i)
    {
      if (value == allowed_values[i])
      {
        ok = true;
        break;
      }
    }
    if (!ok)
    {
      std::stringstream message;
      message << "Illegal value for parameter. Allowed values are: ";
      for (uint i = 0; i < allowed_values.size() - 1; ++i)
      {
        message << "\"" << allowed_values[i] << "\", ";
      }
      message << "\"" << allowed_values[allowed_values.size() - 1] << "\".";
      error(message.str());
    }
  }
  
  // Set value
  this->value = value;
}
//-----------------------------------------------------------------------------
void StringValue::set_range(const std::vector<std::string>& allowed_values)
{
  this->allowed_values = allowed_values;
}
//-----------------------------------------------------------------------------
StringValue::operator std::string() const
{
  return value;
}
//-----------------------------------------------------------------------------
std::string StringValue::type_str() const
{
  return "string";
}
//-----------------------------------------------------------------------------
