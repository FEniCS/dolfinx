// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
ParameterList::ParameterList() : list(DOLFIN_PARAMSIZE), _empty(true)
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
  
  // Add the parameter to the list (resize if necessary)
  if ( !list.add(p) ) {
    list.resize(2*list.size());
    list.add(p);
  }
  
  _empty = false;
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
  int index = getIndex(identifier);

  if ( index >= 0 ) {
	 list(index).set(identifier, aptr);
	 return;
  }

  // Couldn't find the parameter
  dolfin_error1("Unknown parameter \"%s\".", identifier);
}
//----------------------------------------------------------------------------
Parameter ParameterList::get(const char *identifier)
{
  int index = getIndex(identifier);
  
  if ( index >= 0 )
    return list(index);
  
  // Couldn't find the parameter
  dolfin_error1("Unknown parameter \"%s\".", identifier);
}
//----------------------------------------------------------------------------
bool ParameterList::changed(const char *identifier)
{
  int index = getIndex(identifier);

  if ( index >= 0 )
	 return list(index).changed();

  dolfin_error1("Unknown parameter \"%s\".", identifier);
}
//----------------------------------------------------------------------------
bool ParameterList::empty()
{
  return _empty;
}
//----------------------------------------------------------------------------
int ParameterList::getIndex(const char *identifier)
{
  for (ShortList<Parameter>::Iterator it = list.begin(); !it.end(); ++it)
    if ( it->matches(identifier) )
      return it.index();
  
  return -1;
}
//----------------------------------------------------------------------------
