// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005-12-20

#include <string>
#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterList.h>

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
  if ( def(key) )
    dolfin_error1("Unable to add parameter \"%s\" (already defined).",
		  key.c_str());

  parameters.insert(pair(key, value));
}
//-----------------------------------------------------------------------------
void ParameterList::set(std::string key, Parameter value)
{
  iterator p = parameters.find(key);

  if ( p == parameters.end() )
    dolfin_error1("Unknown parameter \"%s\".", key.c_str());

  p->second = value;
}
//-----------------------------------------------------------------------------
Parameter ParameterList::get(std::string key) const
{
  const_iterator p = parameters.find(key);

  if ( p == parameters.end() )
    dolfin_error1("Unknown parameter \"%s\".", key.c_str());
  
  return p->second;
}
//-----------------------------------------------------------------------------
bool ParameterList::def(std::string key) const
{
  return parameters.find(key) != parameters.end();
}
//-----------------------------------------------------------------------------
