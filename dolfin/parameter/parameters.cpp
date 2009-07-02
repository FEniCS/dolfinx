// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
// Modified by Benjamin Kehlet, 2009
// Modified by Ola Skavhaug, 2009
//
// First added:  2005-12-19
// Last changed: 2009-04-30

#include <dolfin/log/LogManager.h>
#include "ParameterSystem.h"
#include "parameters.h"

//-----------------------------------------------------------------------------
dolfin::Parameter dolfin::dolfin_get(std::string key)
{
  return ParameterSystem::parameters.get(key);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_set(std::string key, Parameter value)
{
  ParameterSystem::parameters.set(key, value);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_add(std::string key, Parameter value)
{
  ParameterSystem::parameters.add(key, value);
}
//-----------------------------------------------------------------------------
bool dolfin::dolfin_changed(std::string key)
{
  return ParameterSystem::parameters.changed(key);
}
//-----------------------------------------------------------------------------
