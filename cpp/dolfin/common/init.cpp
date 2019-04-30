// Copyright (C) 2005-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "init.h"
#include "SubSystemsManager.h"
#define LOGURU_WITH_STREAMS 1
#include <dolfin/common/loguru.hpp>

//-----------------------------------------------------------------------------
void dolfin::init(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  LOG_S(INFO) << "Initializing DOLFIN version" << DOLFIN_VERSION;
  common::SubSystemsManager::init_petsc(argc, argv);
}
//-----------------------------------------------------------------------------
