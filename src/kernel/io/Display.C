// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Display.hh>

//-----------------------------------------------------------------------------
int Display::StringLength(const char *string)
{
  if ( !string )
	 return 0;

  int i=0;
  for (i=0;string[i];i++);

  return i;
}
//-----------------------------------------------------------------------------
