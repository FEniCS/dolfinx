// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-18
// Last changed: 2005-12-21

#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterValue.h>

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
const ParameterValue& ParameterValue::operator= (int value)
{
  dolfin_error1("Cannot assign int value to parameter of type %s.",
  		type().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const ParameterValue& ParameterValue::operator= (real value)
{
  dolfin_error1("Cannot assign real value to parameter of type %s.",
		type().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const ParameterValue& ParameterValue::operator= (bool value)
{
  dolfin_error1("Cannot assign bool value to parameter of type %s.",
		type().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const ParameterValue& ParameterValue::operator= (std::string value)
{
  dolfin_error1("Cannot assign string value to parameter of type %s.",
		type().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
const ParameterValue& ParameterValue::operator= (uint value)
{
  dolfin_error1("Cannot assign uint value to parameter of type %s.",
		type().c_str());
  return *this;
}
//-----------------------------------------------------------------------------
ParameterValue::operator int() const
{
  cout << "Halla eller, fel typ" << endl;
  //dolfin_error1("Unable to convert parameter of type %s to int.",
  //		type().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
ParameterValue::operator real() const
{
  dolfin_error1("Unable to convert parameter of type %s to real.",
		type().c_str());
  return 0.0;
}
//-----------------------------------------------------------------------------
ParameterValue::operator bool() const
{
  dolfin_error1("Unable to convert parameter of type %s to bool.",
		type().c_str());
  return false;
}
//-----------------------------------------------------------------------------
ParameterValue::operator std::string() const
{
  dolfin_error1("Unable to convert parameter of type %s to string.",
		type().c_str());
  return "";
}
//-----------------------------------------------------------------------------
ParameterValue::operator dolfin::uint() const
{
  dolfin_error1("Unable to convert parameter of type %s to uint.",
		type().c_str());
  return 0;
}
//-----------------------------------------------------------------------------
