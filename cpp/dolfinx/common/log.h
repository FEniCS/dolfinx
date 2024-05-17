// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <spdlog/spdlog.h>

namespace dolfinx
{

/// @brief Optional initialisation of the logging backend.
///
/// The log verbosity can be controlled from the command line using
/// `SPDLOG_LEVEL=<level>`, where `<level>` is info, warn, debug, etc.
///
/// The full `spdlog` API can be used in applications to control the log
/// system. See https://github.com/gabime/spdlog for the spdlog
/// documentation.
///
/// @param[in] argc Number of command line arguments.
/// @param[in] argv Command line argument vector.
void init_logging(int argc, char* argv[]);

} // namespace dolfinx
