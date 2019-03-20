// Copyright (C) 2003-2005 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "LogManager.h"

// Initialise static data
// FIXME : Logger singleton is initialised here on the first call to logger()
// to avoid "static initialisazation order fiasco". Logger's destructor
// may therefore never be called.

dolfin::log::Logger& dolfin::log::LogManager::logger()
{
  // NB static - this only allocates a new Logger on the first call to logger()
  static dolfin::log::Logger* lg = new (dolfin::log::Logger);
  return *lg;
}
