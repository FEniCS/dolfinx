// Copyright (C) 2010 Garth N. Wells, 2015 Jan Blechta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <optional>
#include <stdexcept>
#include "TimeLogManager.h"
#include "TimeLogger.h"
#include "Timer.h"

using namespace dolfinx;
using namespace dolfinx::common;

//-----------------------------------------------------------------------------
Timer::Timer(std::optional<std::string> task) : _task("")
{
  if (task)
  {
    _task = std::move(task.value());
  }
}
//-----------------------------------------------------------------------------
Timer::~Timer()
{
  if (!_timer.is_stopped())
    stop();
}
//-----------------------------------------------------------------------------
void Timer::start() { _timer.start(); }
//-----------------------------------------------------------------------------
void Timer::resume()
{
  if (!_task.empty())
  {
    throw std::runtime_error(
        "Resuming is not well-defined for logging timer. Only "
        "non-logging timer can be resumed");
  }
  _timer.resume();
}
//-----------------------------------------------------------------------------
double Timer::stop()
{
  _timer.stop();
  const auto [wall, user, system] = this->elapsed();
  if (!_task.empty())
    TimeLogManager::logger().register_timing(_task, wall, user, system);
  return wall;
}
//-----------------------------------------------------------------------------
std::array<double, 3> Timer::elapsed() const
{
  const boost::timer::cpu_times elapsed = _timer.elapsed();
  const double wall = static_cast<double>(elapsed.wall) * 1e-9;
  const double user = static_cast<double>(elapsed.user) * 1e-9;
  const double system = static_cast<double>(elapsed.system) * 1e-9;
  return {wall, user, system};
}
//-----------------------------------------------------------------------------
