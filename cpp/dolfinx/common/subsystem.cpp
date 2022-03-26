// Copyright (C) 2008-2020 Garth N. Wells, Anders Logg, Jan Blechta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "subsystem.h"
#include <dolfinx/common/log.h>
#include <iostream>
#include <mpi.h>
#include <petscsys.h>
#include <string>
#include <vector>

#ifdef HAS_SLEPC
#include <slepcsys.h>
#endif

using namespace dolfinx::common;

//-----------------------------------------------------------------------------
void subsystem::init_logging(int argc, char* argv[])
{
  loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

#ifndef NDEBUG
  loguru::SignalOptions signals;
#else
  loguru::SignalOptions signals = loguru::SignalOptions::none();
#endif

  loguru::Options options = {"-dolfinx_loglevel", "main", signals};

  // Make a copy of argv, as loguru may modify it
  std::vector<char*> argv_copy;
  for (int i = 0; i < argc; ++i)
    argv_copy.push_back(argv[i]);
  argv_copy.push_back(nullptr);

  loguru::init(argc, argv_copy.data(), options);
}
//-----------------------------------------------------------------------------
