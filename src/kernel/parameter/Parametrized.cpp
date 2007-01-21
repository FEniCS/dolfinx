// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-19
// Last changed: 2006-06-02

#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterSystem.h>
#include <dolfin/Parametrized.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Parametrized::Parametrized() : parent(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Parametrized::~Parametrized()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Parametrized::add(std::string key, Parameter value)
{
  parameters.add(key, value);
}
//-----------------------------------------------------------------------------
void Parametrized::set(std::string key, Parameter value)
{
  if ( !def(key) )
    parameters.add(key, value);
  else
    parameters.set(key, value);

  readParameters();
}
//-----------------------------------------------------------------------------
void Parametrized::set(std::string key, const Parametrized& parent)
{
  // Check that key is "parent"
  if ( !(key == "parent") )
    dolfin_error1("Illegal value for parameter \"%s\".", key.c_str());

  // Check if we already have a parent
  if ( this->parent )
    dolfin_error("Local paramater database can only have one parent.");

  // Check that parent is not itself
  if ( this == &parent )
    dolfin_error("Local parameter database cannot be its own parent.");

  // Set parent
  this->parent = &parent;
}
//-----------------------------------------------------------------------------
Parameter Parametrized::get(std::string key) const
{
  // First check local database
  if ( def(key) )
    return parameters.get(key);

  // Check parent if any
  if ( parent )
    return parent->get(key);

  // Fall back on global database
  return dolfin::get(key);
}
//-----------------------------------------------------------------------------
bool Parametrized::def(std::string key) const
{
  return parameters.def(key);
}
//-----------------------------------------------------------------------------
void Parametrized::readParameters()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
