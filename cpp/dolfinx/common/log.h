// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <glog/logging.h>
#include <glog/stl_logging.h>

#pragma once

namespace dolfinx
{

/// @brief Optional initialisation of the logging backend.
///
/// The log verbosity can be controlled from the command line using
/// `--minloglevel=<level>`, where `<level>` is an integer.
/// Increasing values increase verbosity.
///
/// The full `glog` API can be used in applications to control the log
/// system. See https://github.com/google/glog/ for the glog
/// documentation.
///
/// @param[in] argc Number of command line arguments.
/// @param[in] argv Command line argument vector.
void init_logging(int argc, char* argv[]);

} // namespace dolfinx
