// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/GenericLogger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericLogger::GenericLogger()
{
  level = 0;
}
//-----------------------------------------------------------------------------
void GenericLogger::start()
{
  level++;
}
//-----------------------------------------------------------------------------
void GenericLogger::end()
{
  level--;
  
  if ( level < 0 )
    level = 0;
}
//-----------------------------------------------------------------------------
