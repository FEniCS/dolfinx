// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/constants.h>
#include <dolfin/init.h>

//-----------------------------------------------------------------------------
void dolfin::dolfin_init(int argc, char* argv[])
{
  dolfin_info("Initializing DOLFIN version %s.", DOLFIN_VERSION);

  PETScManager::init(argc, argv);
}
//-----------------------------------------------------------------------------
