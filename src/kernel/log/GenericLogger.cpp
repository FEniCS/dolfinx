// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-03-13
// Last changed: 2005

#include <stdio.h>

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
