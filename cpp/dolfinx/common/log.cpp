// Copyright (C) 2019-2026 Chris Richardson and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "log.h"
#include <spdlog/cfg/argv.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>

constexpr std::string logger_name = "dolfinx";

//-----------------------------------------------------------------------------
void dolfinx::init_logging(int argc, char* argv[])
{
  spdlog::stdout_color_mt(logger_name);

  // Initialise to level::warn, can be overridden later
  logger().set_level(spdlog::level::warn);
  spdlog::cfg::load_argv_levels(argc, argv);
}
//-----------------------------------------------------------------------------
auto dolfinx::logger() -> spdlog::logger&
{
  std::shared_ptr<spdlog::logger> ptr = spdlog::get(logger_name);
  if (!ptr)
  {
    throw std::runtime_error("Logger not available. dolfinx::init_logging "
                             "needs to be invoked before.");
  }
  return *ptr;
}
//-----------------------------------------------------------------------------
