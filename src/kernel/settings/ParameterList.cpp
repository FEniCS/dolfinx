// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005-12-08

#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/utsname.h>
#include <stdlib.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterList.h>

using namespace dolfin;

//----------------------------------------------------------------------------
ParameterList::ParameterList()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void ParameterList::add(Parameter::Type type, const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);
 
  add_aptr(type, identifier, aptr);
  
  va_end(aptr);
}
//----------------------------------------------------------------------------
void ParameterList::add_aptr(Parameter::Type type, const char *identifier,
			     va_list aptr)
{
  // Create the parameter
  Parameter p(type, identifier, aptr);
  
  // Add the parameter to the list
  parameters[identifier] = p;
}
//----------------------------------------------------------------------------
void ParameterList::set(const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);
  
  set_aptr(identifier,aptr);
  
  va_end(aptr);
}
//----------------------------------------------------------------------------
void ParameterList::set_aptr(const char *identifier, va_list aptr)
{
  ParameterIterator p = parameters.find(identifier);

  if ( p == parameters.end() )
    dolfin_error1("Unknown parameter \"%s\".", identifier);

  p->second.set(identifier, aptr);
}
//----------------------------------------------------------------------------
Parameter ParameterList::get(const char *identifier)
{
  ParameterIterator p = parameters.find(identifier);
  
  if ( p == parameters.end() )
    dolfin_error1("Unknown parameter \"%s\".", identifier);
  
  return p->second;
}
//----------------------------------------------------------------------------
bool ParameterList::changed(const char *identifier)
{
  ParameterIterator p = parameters.find(identifier);

  if ( p == parameters.end() )
    dolfin_error1("Unknown parameter \"%s\".", identifier);

  return p->second.changed();
}
//----------------------------------------------------------------------------
bool ParameterList::empty()
{
  return parameters.empty();
}
//----------------------------------------------------------------------------
