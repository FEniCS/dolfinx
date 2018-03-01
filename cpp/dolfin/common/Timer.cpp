// Copyright (C) 2010 Garth N. Wells, 2015 Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Timer.h"
#include <dolfin/log/LogManager.h>
#include <dolfin/parameter/GlobalParameters.h>

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------------
Timer::Timer() : _task("")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Timer::Timer(std::string task) : _task(task)
{
  // Do nothing
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
  if (_task.size() > 0)
  {
    log::dolfin_error("Timer.cpp", "resume timing",
                 "Resuming is not well-defined for logging timer. "
                 "Only non-logging timer can be resumed");
  }
  _timer.resume();
}
//-----------------------------------------------------------------------------
double Timer::stop()
{
  _timer.stop();
  const auto elapsed = this->elapsed();
  if (_task.size() > 0)
    log::LogManager::logger().register_timing(_task, elapsed);
  return std::get<0>(elapsed);
}
//-----------------------------------------------------------------------------
std::tuple<double, double, double> Timer::elapsed() const
{
  const auto elapsed = _timer.elapsed();
  const double wall = static_cast<double>(elapsed.wall) * 1e-9;
  const double user = static_cast<double>(elapsed.user) * 1e-9;
  const double system = static_cast<double>(elapsed.system) * 1e-9;
  return std::make_tuple(wall, user, system);
}
//-----------------------------------------------------------------------------
