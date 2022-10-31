// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "log.h"
#include "loguru.cpp"
#include <vector>

//-----------------------------------------------------------------------------
void dolfinx::init_logging(int argc, char* argv[])
{
  loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

#ifdef NDEBUG
  loguru::SignalOptions signals = loguru::SignalOptions::none();
#else
  loguru::SignalOptions signals;
#endif

  loguru::Options options = {"-dolfinx_loglevel", "main", signals};

  // Make a copy of argv, as loguru may modify it
  std::vector<char*> argv_copy(argv, argv + argc);
  argv_copy.push_back(nullptr);

  loguru::init(argc, argv_copy.data(), options);
}
//-----------------------------------------------------------------------------
