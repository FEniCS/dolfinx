// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-02-13
// Last changed: 2006-05-07

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/constants.h>
#include <dolfin/init.h>

//-----------------------------------------------------------------------------
void dolfin::dolfin_init(int argc, char* argv[])
{
  message("Initializing DOLFIN version %s.", DOLFIN_VERSION);
  
#ifdef HAVE_PETSC_H
  PETScManager::init(argc, argv);
#endif
}
//-----------------------------------------------------------------------------
