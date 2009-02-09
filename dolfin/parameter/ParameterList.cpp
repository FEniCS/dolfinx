// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet
//
// First added:  2003-05-06
// Last changed: 2008-11-18

#include <string>
#include <dolfin/log/dolfin_log.h>
#include "ParameterList.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ParameterList::ParameterList()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ParameterList::~ParameterList()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ParameterList::add(std::string key, Parameter value)
{
  if ( defined(key) )
    error("Unable to add parameter \"%s\" (already defined).",
		  key.c_str());

  parameters.insert(pair(key, value));
}
//-----------------------------------------------------------------------------
void ParameterList::set(std::string key, Parameter value)
{
  iterator p = parameters.find(key);

  if ( p == parameters.end() )
    error("Unknown parameter \"%s\".", key.c_str());

  p->second = value;
  p->second._changed = true;
}
//-----------------------------------------------------------------------------
Parameter ParameterList::get(std::string key) const
{
  const_iterator p = parameters.find(key);

  if ( p == parameters.end() )
    error("Unknown parameter \"%s\".", key.c_str());
  
  return p->second;
}
//-----------------------------------------------------------------------------
bool ParameterList::defined(std::string key) const
{
  return parameters.find(key) != parameters.end();
}
//-----------------------------------------------------------------------------
bool ParameterList::changed(std::string key) const
{
  return get(key)._changed;
}
//-----------------------------------------------------------------------------
