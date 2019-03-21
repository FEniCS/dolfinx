// Copyright (C) 2005-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <spdlog/spdlog.h>

#include "SubSystemsManager.h"
#include "init.h"

//-----------------------------------------------------------------------------
void dolfin::init(int argc, char* argv[])
{
  spdlog::info("Initializing DOLFIN version {}", DOLFIN_VERSION);
  common::SubSystemsManager::init_petsc(argc, argv);
}
//-----------------------------------------------------------------------------
