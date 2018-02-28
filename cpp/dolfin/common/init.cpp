// Copyright (C) 2005-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "init.h"
#include "SubSystemsManager.h"
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>

//-----------------------------------------------------------------------------
void dolfin::init(int argc, char* argv[])
{
  log::log(PROGRESS, "Initializing DOLFIN version %s.", DOLFIN_VERSION);

#ifdef HAS_PETSC
  common::SubSystemsManager::init_petsc(argc, argv);
#endif
}
//-----------------------------------------------------------------------------
