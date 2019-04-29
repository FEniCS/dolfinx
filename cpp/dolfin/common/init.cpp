// Copyright (C) 2005-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// #include <glog/glog.h>
#include "init.h"
#include "SubSystemsManager.h"

//-----------------------------------------------------------------------------
void dolfin::init(int argc, char* argv[])
{
  // glog::info("Initializing DOLFIN version {}", DOLFIN_VERSION);
  common::SubSystemsManager::init_petsc(argc, argv);
}
//-----------------------------------------------------------------------------
