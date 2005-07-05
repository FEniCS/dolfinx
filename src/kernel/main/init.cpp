// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-02-13
// Last changed: 2005

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
