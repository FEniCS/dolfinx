// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "log.h"

#include <gflags/gflags.h>
#include <vector>

//-----------------------------------------------------------------------------
void dolfinx::init_logging(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
}
//-----------------------------------------------------------------------------
