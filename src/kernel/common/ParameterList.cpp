#include <iostream>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/utsname.h>
#include <stdlib.h>
#include <dolfin/ParameterList.h>

using namespace dolfin;

// Initialise static data
ShortList<Parameter> ParameterList::list(DOLFIN_PARAMSIZE);
bool ParameterList::_empty(true);

//----------------------------------------------------------------------------
void ParameterList::add(Parameter::Type type, const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);
  
  // Create the parameter
  Parameter p(type, identifier, aptr);
  
  // Add the parameter to the list (resize if necessary)
  if ( !list.add(p) ) {
	 list.resize(2*list.size());
	 list.add(p);
  }
  
  va_end(aptr);

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

  if ( index >= 0 ){
	 list(index).set(identifier, aptr);
	 return;
  }

  // Couldn't find the parameter
  std::cout << "Warning: Trying to set value of unknown parameter \"" << identifier << "\"." << std::endl;
}
//----------------------------------------------------------------------------
void ParameterList::get(const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);

  get_aptr(identifier,aptr);

  va_end(aptr);
}
//----------------------------------------------------------------------------
void ParameterList::get_aptr(const char *identifier, va_list aptr)
{
  int index = getIndex(identifier);
  
  if ( index >= 0 ){
	 list(index).get(aptr);
	 return;
  }
  
  // Couldn't find the parameter
  std::cout << "Warning: Unknown parameter \"" << identifier << "\"" << std::endl;
}
//----------------------------------------------------------------------------
bool ParameterList::changed(const char *identifier)
{
  int index = getIndex(identifier);

  if ( index >= 0 )
	 return list(index).changed();

  std::cout << "Error: Status for unknown parameter <" << identifier << "> not available." << std::endl;
  exit(1);
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
