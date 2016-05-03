// Copyright (C) 2010 Garth N. Wells, 2015 Jan Blechta
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-09-08
// Last changed:

#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/log/LogManager.h>
#include "Timer.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Timer::Timer() : _task("")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Timer::Timer(std::string task) : _task("")
{
  const std::string prefix = parameters["timer_prefix"];
  _task = prefix + task;
}
//-----------------------------------------------------------------------------
Timer::~Timer()
{
 if (!_timer.is_stopped())
   stop();
}
//-----------------------------------------------------------------------------
void Timer::start()
{
  _timer.start();
}
//-----------------------------------------------------------------------------
void Timer::resume()
{
  if (_task.size() > 0)
    dolfin_error("Timer.cpp",
                 "resume timing",
                 "Resuming is not well-defined for logging timer. "
                 "Only non-logging timer can be resumed");
  _timer.resume();
}
//-----------------------------------------------------------------------------
double Timer::stop()
{
  _timer.stop();
  const auto elapsed = this->elapsed();
  if (_task.size() > 0)
    LogManager::logger().register_timing(_task, elapsed);
  return std::get<0>(elapsed);
}
//-----------------------------------------------------------------------------
std::tuple<double, double, double> Timer::elapsed() const
{
  const auto elapsed = _timer.elapsed();
  const double wall   = static_cast<double>(elapsed.wall  ) * 1e-9;
  const double user   = static_cast<double>(elapsed.user  ) * 1e-9;
  const double system = static_cast<double>(elapsed.system) * 1e-9;
  return std::make_tuple(wall, user, system);
}
//-----------------------------------------------------------------------------
