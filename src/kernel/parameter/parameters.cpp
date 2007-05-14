// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2005-12-19
// Last changed: 2007-05-14

#include <dolfin/LogManager.h>
#include <dolfin/ParameterSystem.h>
#include <dolfin/parameters.h>

//-----------------------------------------------------------------------------
void dolfin::add(std::string key, Parameter value)
{
  ParameterSystem::parameters.add(key, value);
}
//-----------------------------------------------------------------------------
void dolfin::set(std::string key, Parameter value)
{
  // Special cases: pass on to log system
  if (key == "debug level")
  {
    LogManager::logger.setDebugLevel(value);
    return;
  }
  else if (key == "output destination")
  {
    LogManager::logger.setOutputDestination(value);
    return;
  }

  ParameterSystem::parameters.set(key, value);
}
//-----------------------------------------------------------------------------
dolfin::Parameter dolfin::get(std::string key)
{
  return ParameterSystem::parameters.get(key);
}
//-----------------------------------------------------------------------------
