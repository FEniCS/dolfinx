// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

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
  list.push_back(p);
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
  Parameter* p = find(identifier);
  
  if ( p ) {
    p->set(identifier, aptr);
    return;
  }

  // Couldn't find the parameter
  dolfin_error1("Unknown parameter \"%s\".", identifier);
}
//----------------------------------------------------------------------------
Parameter ParameterList::get(const char *identifier)
{
  Parameter* p = find(identifier);
  
  if ( p )
    return *p;
  
  // Couldn't find the parameter
  dolfin_error1("Unknown parameter \"%s\".", identifier);

  return *p;
}
//----------------------------------------------------------------------------
bool ParameterList::changed(const char *identifier)
{
  Parameter* p = find(identifier);

  if ( p )
    return p->changed();

  dolfin_error1("Unknown parameter \"%s\".", identifier);

  return *p;
}
//----------------------------------------------------------------------------
bool ParameterList::empty()
{
  return list.empty();
}
//----------------------------------------------------------------------------
Parameter* ParameterList::find(const char *identifier)
{
  for (List<Parameter>::iterator p = list.begin(); p != list.end(); ++p)
    if ( p->matches(identifier) )
      return &(*p);

  return 0;
}
//----------------------------------------------------------------------------
