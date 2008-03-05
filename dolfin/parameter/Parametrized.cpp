// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-19
// Last changed: 2008-02-11

#include <dolfin/log/dolfin_log.h>
#include "parameters.h"
#include "ParameterSystem.h"
#include "Parametrized.h"

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
  if ( !has(key) )
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
    error("Illegal value for parameter \"%s\".", key.c_str());

  // Check if we already have a parent
  if ( this->parent )
    error("Local paramater database can only have one parent.");

  // Check that parent is not itself
  if ( this == &parent )
    error("Local parameter database cannot be its own parent.");

  // Set parent
  this->parent = &parent;
}
//-----------------------------------------------------------------------------
Parameter Parametrized::get(std::string key) const
{
  // First check local database
  if ( has(key) )
    return parameters.get(key);

  // Check parent if any
  if ( parent )
    return parent->get(key);

  // Fall back on global database
  return dolfin::dolfin_get(key);
}
//-----------------------------------------------------------------------------
bool Parametrized::has(std::string key) const
{
  return parameters.defined(key);
}
//-----------------------------------------------------------------------------
void Parametrized::readParameters()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
