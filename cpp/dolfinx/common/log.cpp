// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "log.h"
#include <spdlog/cfg/argv.h>
#include <vector>

//-----------------------------------------------------------------------------
void dolfinx::init_logging(int argc, char* argv[])
{
  // Initialise to level::warn, can be overridden later
  spdlog::set_level(spdlog::level::warn);
  spdlog::cfg::load_argv_levels(argc, argv);
}
//-----------------------------------------------------------------------------
