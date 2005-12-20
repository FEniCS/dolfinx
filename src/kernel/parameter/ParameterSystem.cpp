// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-19
// Last changed: 2005-12-19

#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterSystem.h>

// Initialize the global parameter database
dolfin::ParameterSystem dolfin::ParameterSystem::parameters;

using namespace dolfin;

//-----------------------------------------------------------------------------
ParameterSystem::ParameterSystem() : ParameterList()
{
  dolfin_info("Initializing DOLFIN parameter system.");

  // Include default values for parameters
#include <dolfin/DefaultParameters.h>
}
//-----------------------------------------------------------------------------
void dolfin::add(std::string key, Parameter value)
{
  ParameterSystem::parameters.add(key, value);
}
//-----------------------------------------------------------------------------
void dolfin::set(std::string key, Parameter value)
{
  ParameterSystem::parameters.set(key, value);
}
//-----------------------------------------------------------------------------
dolfin::Parameter dolfin::get(std::string key)
{
  return ParameterSystem::parameters.get(key);
}
//-----------------------------------------------------------------------------
