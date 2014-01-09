// Copyright (C) 2010 Garth N. Wells
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
#include "timing.h"
#include "Timer.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Timer::Timer(std::string task) : _task(""), t(time()), stopped(false)
{
  const std::string prefix = parameters["timer_prefix"];
  _task = prefix + task;
}
//-----------------------------------------------------------------------------
Timer::~Timer()
{
 if (!stopped)
   stop();
}
//-----------------------------------------------------------------------------
void Timer::start()
{
  t = time();
  stopped = false;
}
//-----------------------------------------------------------------------------
double Timer::stop()
{
  t = time() - t;
  LogManager::logger.register_timing(_task, t);
  stopped = true;
  return t;
}
//-----------------------------------------------------------------------------
double Timer::value() const
{
  return t;
}
//-----------------------------------------------------------------------------
