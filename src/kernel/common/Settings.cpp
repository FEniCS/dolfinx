// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <stdarg.h>
#include <dolfin/Settings.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void Settings::get(const char *identifier, ...)
{
  // Make sure that we call the constructor
  if ( empty() )
	 Settings::Settings();
  
  va_list aptr;
  va_start(aptr,identifier);

  get_aptr(identifier,aptr);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Settings::set(const char *identifier, ...)
{
  // Make sure that we call the constructor
  if ( empty() )
	 Settings::Settings();
  
  va_list aptr;
  va_start(aptr,identifier);

  get_aptr(identifier,aptr);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
