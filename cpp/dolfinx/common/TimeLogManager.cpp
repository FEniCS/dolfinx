// Copyright (C) 2003-2005 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TimeLogManager.h"

// Initialise static data
// FIXME : Logger singleton is initialised here on the first call to logger()
// to avoid "static initialisation order fiasco". Logger's destructor
// may therefore never be called.

dolfinx::common::TimeLogger& dolfinx::common::TimeLogManager::logger()
{
  // NB static - this only allocates a new Logger on the first call to logger()
  static dolfinx::common::TimeLogger* lg = new (dolfinx::common::TimeLogger);
  return *lg;
}
