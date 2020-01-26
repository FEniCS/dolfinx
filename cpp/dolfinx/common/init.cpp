// Copyright (C) 2005-2011 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "init.h"
#include "SubSystemsManager.h"
#include <dolfinx/common/log.h>

//-----------------------------------------------------------------------------
void dolfinx::init(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  LOG(INFO) << "Initializing DOLFINX version" << DOLFINX_VERSION;
  common::SubSystemsManager::init_petsc(argc, argv);
}
//-----------------------------------------------------------------------------
